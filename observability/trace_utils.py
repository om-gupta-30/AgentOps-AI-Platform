"""
Trace utilities (LangSmith/Langfuse compatible).

Why tracing is optional:
- Observability is an operational concern; core agent behavior must work in dev/test/offline modes.
- Teams may enable/disable or switch backends per environment (local vs staging vs prod).

Why tracing should never break core logic:
- If missing credentials or a vendor outage causes exceptions, you don't want your API/agent runs to fail.
- Tracing should be best-effort: emit signals when available, otherwise degrade to no-ops.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Literal, Optional


Backend = Literal["langsmith", "langfuse", "none"]

_ACTIVE_TRACE: ContextVar[Any | None] = ContextVar("_ACTIVE_TRACE", default=None)
_ACTIVE_BACKEND: ContextVar[Backend] = ContextVar("_ACTIVE_BACKEND", default="none")


def _select_backend(explicit: Optional[Backend] = None) -> Backend:
    """
    Decide which backend to use.

    Selection order:
    - explicit argument
    - OBSERVABILITY_BACKEND env var (langsmith/langfuse/none)
    - default: none
    """

    if explicit:
        return explicit

    import os

    value = os.getenv("OBSERVABILITY_BACKEND", "none").strip().lower()
    if value in ("langsmith", "langfuse", "none"):
        return value  # type: ignore[return-value]
    return "none"


@contextmanager
def traced(
    name: str,
    metadata: Optional[dict[str, Any]] = None,
    backend: Optional[Backend] = None,
) -> Iterator[Any]:
    """
    Start a trace/span and yield a handle.

    - Compatible with LangSmith and Langfuse scaffolding modules.
    - Safe-by-default: missing deps/credentials will not crash; it becomes a no-op.

    The yielded handle supports:
    - `.span(name, metadata=...)` as a nested context manager (no-op in scaffolding)
    - `.end()` (no-op in scaffolding)
    """

    selected = _select_backend(backend)
    meta = metadata or {}

    active = _ACTIVE_TRACE.get()
    active_backend = _ACTIVE_BACKEND.get()

    # If there is already an active trace in this execution context, create a span under it.
    # This enables a single request-level trace with node-level spans, without needing to
    # thread trace handles through every function signature.
    if active is not None and selected != "none" and (selected == active_backend):
        try:
            with active.span(name, metadata=meta) as span:
                yield span
            return
        except Exception:
            # Best-effort only: never let observability break core logic.
            pass

    try:
        if selected == "langsmith":
            from observability.langsmith import create_trace

            trace = create_trace(name=name, metadata=meta)
        elif selected == "langfuse":
            from observability.langfuse import create_trace

            trace = create_trace(name=name, metadata=meta)
        else:
            # Explicit no tracing.
            trace = None
    except Exception:
        # Best-effort only: never let observability break core logic.
        trace = None

    if trace is None:
        # Minimal no-op object with span/end semantics (duck-typed).
        class _NoOp:
            @contextmanager
            def span(self, _name: str, metadata: Optional[dict[str, Any]] = None) -> Iterator["_NoOp"]:
                yield self

            def end(self) -> None:
                return

        trace = _NoOp()

    try:
        token = _ACTIVE_TRACE.set(trace)
        backend_token = _ACTIVE_BACKEND.set(selected)
        yield trace
    finally:
        # Best-effort end; never raise.
        try:
            end = getattr(trace, "end", None)
            if callable(end):
                end()
        except Exception:
            pass
        finally:
            try:
                _ACTIVE_TRACE.reset(token)
                _ACTIVE_BACKEND.reset(backend_token)
            except Exception:
                return

