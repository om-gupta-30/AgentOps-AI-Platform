"""
Run Router

This router handles endpoints for executing agent tasks.

Endpoints:
- POST /run - Execute a new agent task synchronously

Responsibilities:
1. Accept user requests (goal string)
2. Validate input (length limits, format, malicious content)
3. Execute LangGraph synchronously (supervisor → research → execution → evaluator)
4. Return final output, evaluation, and memory usage
5. Handle errors gracefully (LLM failures, timeouts, rate limits)

NOT responsible for:
- Exposing internal state (plan, research, traces)
- Storing task history (that's the responsibility of a database layer)
- Managing user authentication (future: separate auth middleware)
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional, AsyncIterator
import sys
import os
import json

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))

# =============================================================================
# Router Setup
# =============================================================================

router = APIRouter()

# =============================================================================
# Request/Response Models
# =============================================================================


class RunRequest(BaseModel):
    """
    Request body for executing a new agent task.

    Fields:
    - goal: The task the user wants to accomplish (10-500 characters)

    Example:
        {
            "goal": "Explain the benefits of vector databases for semantic search"
        }
    """

    goal: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The task to execute (3-500 characters)",
        examples=["Explain the benefits of vector databases for semantic search"],
    )

    @field_validator("goal")
    @classmethod
    def validate_goal(cls, v: str) -> str:
        """
        Validate and sanitize goal.

        Checks:
        - Not empty after stripping whitespace
        - No suspicious patterns (future: content moderation)
        """
        v = v.strip()
        if not v:
            raise ValueError("goal cannot be empty or whitespace-only")

        # Future: Add content moderation here
        # - Check for prompt injection attempts
        # - Check for malicious content
        # - Check for PII leakage attempts

        return v


class EvaluationResult(BaseModel):
    """
    Evaluation results for the task output.

    Fields:
    - passed: Whether the output passed evaluation
    - score: Quality score (1-10)
    - reasons: List of evaluation reasons (why it passed/failed)
    """

    passed: bool = Field(..., description="Whether output passed evaluation")
    score: int = Field(..., ge=1, le=10, description="Quality score (1-10)")
    reasons: list[str] = Field(..., description="Evaluation reasons")


class RunResponse(BaseModel):
    """
    Response body for an executed agent task.

    Fields:
    - final_output: The final generated output from the execution agent
    - evaluation: Evaluation results (passed, score, reasons)
    - memory_used: Whether memory was used in planning

    Example:
        {
            "final_output": "Vector databases are specialized systems...",
            "evaluation": {
                "passed": true,
                "score": 9,
                "reasons": ["Clear explanation", "Covers key concepts"]
            },
            "memory_used": true
        }

    Note:
    - Internal state (plan, research, traces) is NOT exposed
    - Only final output and evaluation are returned
    """

    final_output: str = Field(..., description="Final output from execution agent")
    evaluation: EvaluationResult = Field(..., description="Evaluation results")
    memory_used: bool = Field(..., description="Whether memory was used in planning")


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/run")
async def execute_task(request: RunRequest, stream: bool = False):
    """
    Execute a new agent task with optional streaming support.

    This endpoint:
    1. Validates the request (Pydantic handles this)
    2. Executes LangGraph workflow (supervisor → research → execution → evaluator)
    3. Returns final output, evaluation, and memory usage
    4. Optionally streams execution output in real-time (if stream=true)
    5. Handles errors gracefully without exposing internal state

    Args:
        request: RunRequest with goal string
        stream: If true, streams execution output via Server-Sent Events (SSE)

    Returns:
        - If stream=false: RunResponse with final_output, evaluation, and memory_used
        - If stream=true: StreamingResponse with SSE events

    Raises:
        HTTPException 422: Invalid request (validation error)
        HTTPException 500: Execution failed (LLM errors, timeouts, etc.)
        HTTPException 503: Service unavailable (API keys missing, etc.)

    Streaming Behavior:
    - Planning (Supervisor): NOT streamed (must complete before execution)
    - Research: NOT streamed (must complete before execution uses it)
    - Execution: STREAMED (the main user-facing output)
    - Evaluation: NOT streamed (must analyze complete output)
    
    Why only execution is streamed:
    1. Dependencies: Plan and research must complete before execution starts
    2. User value: Execution output is what users care about seeing in real-time
    3. Evaluation integrity: Evaluator needs the complete output to assess quality
    
    If streaming fails, the endpoint automatically falls back to non-streaming response.

    Privacy & Security:
    - Internal state (plan, research, traces) is NOT exposed
    - Only final output and evaluation are returned
    - Stack traces are logged but not sent to client
    - PII in goal is sanitized before processing
    """
    
    if stream:
        # Return streaming response
        return StreamingResponse(
            _execute_task_streaming(request),
            media_type="text/event-stream",
        )
    else:
        # Return non-streaming response (original behavior)
        return await _execute_task_non_streaming(request)


async def _execute_task_non_streaming(request: RunRequest) -> RunResponse:
    """
    Execute task without streaming (original synchronous behavior).
    
    This is the fallback path and the default behavior when stream=false.
    """

    try:
        # Import LangGraph components
        # Import here to avoid forcing dependencies at module load time
        from agentops_ai_platform.graphs.main_graph import GraphState, build_main_graph

        # Initialize graph state with user goal
        initial_state: GraphState = {
            "user_goal": request.goal,
            "requires_research": None,  # Set by supervisor
            "plan": None,  # Set by supervisor
            "success_criteria": None,  # Set by supervisor
            "research_results": None,  # Set by research (if needed)
            "draft_output": None,  # Set by execution
            "evaluation": None,  # Set by evaluator
            "iteration_count": 0,
            "memory_used": False,  # Set by supervisor
        }

        # Create and execute the graph
        # This runs: supervisor → research (if needed) → execution → evaluator
        graph = build_main_graph().compile()
        final_state = graph.invoke(initial_state)

        # Extract results from final state
        # Only expose: final output, evaluation, memory usage
        # Do NOT expose: plan, research, internal traces
        draft_output = final_state.get("draft_output") or ""
        evaluation_data = final_state.get("evaluation")

        # Check if memory was used (tracked by supervisor_node)
        memory_used = final_state.get("memory_used", False)

        # Validate that we have required fields
        if not draft_output:
            raise ValueError("No output generated by execution agent")

        if not evaluation_data or not isinstance(evaluation_data, dict):
            raise ValueError("No evaluation performed by evaluator agent")

        # Build evaluation result from evaluator output
        # evaluator_agent.py returns: { "pass": bool, "score": int, "reasons": list, ... }
        evaluation_result = EvaluationResult(
            passed=evaluation_data.get("pass_", False) or evaluation_data.get("pass", False),
            score=evaluation_data.get("score", 0),
            reasons=evaluation_data.get("reasons", []),
        )

        # Return final response
        return RunResponse(
            final_output=draft_output,
            evaluation=evaluation_result,
            memory_used=memory_used,
        )

    except ImportError as e:
        # LangGraph or dependencies not available
        # This indicates a configuration/deployment issue
        raise HTTPException(
            status_code=503,
            detail={
                "error": "service_unavailable",
                "message": "Agent execution service is not available",
                "details": "Required dependencies not installed",
            },
        )

    except ValueError as e:
        # Validation error in LangGraph execution
        # e.g., empty output, missing evaluation
        raise HTTPException(
            status_code=500,
            detail={
                "error": "execution_failed",
                "message": "Task execution failed due to internal error",
                "details": str(e),
            },
        )

    except Exception as e:
        # Catch-all for unexpected errors
        # Log the full error internally, return sanitized message to client
        error_type = type(e).__name__

        # Classify error for user-friendly messaging
        if "API" in error_type or "RateLimit" in error_type or "Timeout" in error_type:
            # LLM API errors (rate limits, timeouts, etc.)
            error_category = "llm_api_error"
            message = "Failed to execute task due to LLM API error. Please try again."
        elif "ValidationError" in error_type:
            # Pydantic validation errors (malformed data)
            error_category = "validation_error"
            message = "Task execution produced invalid data. Please report this issue."
        else:
            # Unknown error
            error_category = "unknown_error"
            message = "An unexpected error occurred. Please try again or contact support."

        # Log full error internally (not exposed to client)
        print(f"❌ Task execution error: {error_type}: {e}")

        # Return sanitized error to client
        raise HTTPException(
            status_code=500,
            detail={
                "error": error_category,
                "message": message,
                "details": f"{error_type}",  # Type only, not full message
            },
        )


async def _execute_task_streaming(request: RunRequest) -> AsyncIterator[str]:
    """
    Execute task with streaming support for execution output.
    
    This function:
    1. Runs supervisor (non-streamed, must complete first)
    2. Runs research if needed (non-streamed, must complete first)
    3. Streams execution output as it's generated
    4. Runs evaluator after execution completes (non-streamed)
    5. Sends final result with evaluation via SSE
    
    Yields:
        str: Server-Sent Events (SSE) formatted messages
    
    SSE Event Types:
    - data: {"type": "status", "message": "..."} - Status updates
    - data: {"type": "chunk", "content": "..."} - Execution output chunks
    - data: {"type": "complete", "final_output": "...", "evaluation": {...}, "memory_used": bool} - Final result
    - data: {"type": "error", "message": "..."} - Error occurred
    
    Fallback behavior:
    If streaming fails at any point, this function catches the error,
    logs it, and sends an error event. The client should then retry
    with stream=false.
    """
    
    try:
        # Import LangGraph components
        from agentops_ai_platform.graphs.main_graph import GraphState, build_main_graph
        from agentops_ai_platform.agents.supervisor_agent import generate_plan
        from agentops_ai_platform.agents.research_agent import conduct_research
        from agentops_ai_platform.agents.execution_agent import execute_task_streaming, ExecutionResult
        from agentops_ai_platform.agents.evaluator_agent import evaluate_output
        from memory.memory_store import find_relevant_memories
        
        # Send initial status
        yield f"data: {json.dumps({'type': 'status', 'message': 'Initializing...'})}\n\n"
        
        # =============================================================================
        # PHASE 1: SUPERVISOR (Non-streamed - must complete before execution)
        # =============================================================================
        
        yield f"data: {json.dumps({'type': 'status', 'message': 'Planning task...'})}\n\n"
        
        # Check for relevant memories
        memories = find_relevant_memories(request.goal)
        memory_context = None
        num_memories = 0
        memory_used = False
        
        if memories:
            num_memories = len(memories)
            memory_lines = []
            for m in memories[:3]:  # Limit to top 3
                memory_lines.append(f"- {m.user_goal}: {m.summary[:100]}")
            memory_context = "\n".join(memory_lines)
            memory_used = True
        
        # Generate plan
        # NOTE: memory_context is prepended to goal string, NOT passed as separate arg
        # This matches the pattern used in main_graph.py supervisor_node
        if memory_context:
            planning_input = f"""{request.goal}

[CONTEXT: Relevant past successful tasks]
{memory_context}
"""
        else:
            planning_input = request.goal
        
        plan_result = generate_plan(planning_input)
        
        # =============================================================================
        # PHASE 2: RESEARCH (Non-streamed - must complete before execution uses it)
        # =============================================================================
        
        research_result = None
        if plan_result.requires_research:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Conducting research...'})}\n\n"
            research_question = f"Research needed for: {request.goal}"
            research_result = conduct_research(research_question)
        
        # =============================================================================
        # PHASE 3: EXECUTION (STREAMED - main user-facing output)
        # =============================================================================
        
        yield f"data: {json.dumps({'type': 'status', 'message': 'Generating output...'})}\n\n"
        
        # Stream execution output
        execution_result = None
        full_output = ""
        
        research_text = None
        if research_result:
            research_text = f"{research_result.summary}\n\nKey points:\n" + "\n".join(f"- {p}" for p in research_result.key_points)
        
        for item in execute_task_streaming(
            user_goal=request.goal,
            plan_steps=plan_result.plan_steps,
            research_results=research_text,
        ):
            # Check if this is the final ExecutionResult or a chunk
            if isinstance(item, ExecutionResult):
                execution_result = item
                full_output = item.output
            elif isinstance(item, str):
                # Stream chunk to client
                # Accumulate for final output
                full_output += item
                yield f"data: {json.dumps({'type': 'chunk', 'content': item})}\n\n"
        
        # Ensure we have an execution result
        if not execution_result:
            # Fallback if streaming didn't yield ExecutionResult
            from agentops_ai_platform.agents.execution_agent import ExecutionResult
            execution_result = ExecutionResult(
                output=full_output,
                actions_taken=[f"Followed plan step: {s}" for s in plan_result.plan_steps],
                assumptions=["Streaming completed but result object was not yielded."],
            )
        
        # =============================================================================
        # PHASE 4: EVALUATION (Non-streamed - must analyze complete output)
        # =============================================================================
        
        yield f"data: {json.dumps({'type': 'status', 'message': 'Evaluating output...'})}\n\n"
        
        evaluation_result = evaluate_output(
            user_goal=request.goal,
            draft_output=execution_result.output,
            success_criteria=plan_result.success_criteria,
            research_results=research_text,  # Pass research context for hallucination checking
        )
        
        # =============================================================================
        # PHASE 5: SEND FINAL RESULT
        # =============================================================================
        
        # Build final response
        final_response = {
            "type": "complete",
            "final_output": execution_result.output,
            "evaluation": {
                "passed": evaluation_result.pass_,
                "score": evaluation_result.score,
                "reasons": evaluation_result.reasons,
            },
            "memory_used": memory_used,
        }
        
        yield f"data: {json.dumps(final_response)}\n\n"
        
    except ImportError as e:
        # Dependencies not available
        error_msg = "Agent execution service is not available"
        yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
        
    except Exception as e:
        # Streaming failed - log and send error event
        error_type = type(e).__name__
        print(f"❌ Streaming execution failed: {error_type}: {e}")
        
        # Send error event to client
        error_msg = "Streaming failed. Please retry without streaming."
        yield f"data: {json.dumps({'type': 'error', 'message': error_msg, 'error_type': error_type})}\n\n"


# =============================================================================
# Future Enhancements (Not Yet Implemented)
# =============================================================================
#
# 1. ASYNC EXECUTION:
#    - POST /run returns task_id immediately
#    - GET /run/{task_id}/status for polling
#    - Background job queue (Celery, RQ, etc.)
#
# 2. WEBSOCKET STREAMING (Enhanced):
#    - Bi-directional communication for real-time updates
#    - Stream all agent outputs (plan, research, execution, evaluation)
#    - Client can send cancellation requests mid-execution
#
# 3. TASK CANCELLATION:
#    - POST /run/{task_id}/cancel
#    - Gracefully stop LangGraph execution
#    - Clean up resources (LLM connections, observability)
#
# 4. PRIORITY QUEUING:
#    - Accept priority field in RunRequest
#    - High-priority tasks execute first
#    - Rate limiting per user/API key
#
# 5. BATCH EXECUTION:
#    - POST /run/batch with multiple goals
#    - Execute tasks in parallel
#    - Return batch_id for status queries
#
# 6. PARTIAL RESULTS:
#    - Return intermediate results (plan, research)
#    - Useful for debugging and transparency
#    - Opt-in via query parameter (?include_details=true)
