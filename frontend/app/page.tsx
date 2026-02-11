'use client'

import { useState, useEffect } from 'react'
import AgentInput from './components/AgentInput'
import ResultDisplay from './components/ResultDisplay'
import HistoryList from './components/HistoryList'

// =============================================================================
// Types
// =============================================================================

interface Evaluation {
  passed: boolean
  score: number
  reasons: string[]
}

interface ApiResponse {
  final_output: string
  evaluation: Evaluation
  memory_used: boolean
}

interface ErrorResponse {
  error: string
  message: string
  details?: string
}

interface MemoryRecord {
  user_goal: string
  summary: string
  score: number
  created_at: string
}

interface HistoryResponse {
  memories: MemoryRecord[]
  total: number
}

export default function Home() {
  // =============================================================================
  // State Management
  // =============================================================================
  // All state is managed at the page level to coordinate between input and result display

  const [goal, setGoal] = useState('')
  const [loading, setLoading] = useState(false)
  const [streamingStatus, setStreamingStatus] = useState<string | null>(null) // "Planning...", "Generating...", etc.
  const [result, setResult] = useState<string | null>(null)
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null)
  const [memoryUsed, setMemoryUsed] = useState<boolean | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [history, setHistory] = useState<MemoryRecord[]>([])
  const [historyLoading, setHistoryLoading] = useState(true)

  // =============================================================================
  // Load History on Mount
  // =============================================================================
  // Fetch past successful tasks once when the page loads
  // No auto-refresh - keeps network usage minimal

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const response = await fetch('http://localhost:8000/history')
        if (response.ok) {
          const data: HistoryResponse = await response.json()
          setHistory(data.memories)
        } else {
          // History endpoint failed - not critical, just log it
          console.warn('Failed to load history')
        }
      } catch (err) {
        // Network error loading history - not critical
        console.warn('Could not fetch history:', err)
      } finally {
        setHistoryLoading(false)
      }
    }

    fetchHistory()
  }, []) // Empty dependency array = run once on mount

  // =============================================================================
  // API Call Handler with Streaming Support
  // =============================================================================

  const handleRunAgent = async (goalText: string) => {
    // Clear previous results and errors
    setError(null)
    setResult(null)
    setEvaluation(null)
    setMemoryUsed(null)
    setStreamingStatus(null)
    setLoading(true)

    // Try streaming first, fall back to non-streaming if it fails
    const useStreaming = true

    if (useStreaming) {
      try {
        await handleStreamingRequest(goalText)
        return
      } catch (err) {
        // Streaming failed, fall back to non-streaming
        console.warn('Streaming failed, falling back to non-streaming:', err)
        setError(null) // Clear any streaming errors
        setStreamingStatus(null)
      }
    }

    // Non-streaming fallback
    await handleNonStreamingRequest(goalText)
  }

  const handleStreamingRequest = async (goalText: string) => {
    try {
      // =============================================================================
      // Call backend API with streaming enabled
      // =============================================================================
      // Using fetch with ReadableStream to receive Server-Sent Events (SSE)
      // Backend sends events in this format: "data: {json}\n\n"
      
      const response = await fetch('http://localhost:8000/run?stream=true', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ goal: goalText }),
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      if (!response.body) {
        throw new Error('Response body is null')
      }

      // =============================================================================
      // Read streaming response using ReadableStream
      // =============================================================================
      // The response body is a stream of bytes that we decode incrementally
      // SSE events are text-based and separated by newlines
      
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = '' // Accumulates partial lines between reads
      let currentOutput = '' // Accumulates execution output chunks

      while (true) {
        // Read next chunk from stream
        const { done, value } = await reader.read()

        if (done) {
          // Stream completed
          break
        }

        // Decode bytes to text and add to buffer
        buffer += decoder.decode(value, { stream: true })

        // =============================================================================
        // Process complete SSE events (lines)
        // =============================================================================
        // SSE events are separated by newlines (\n\n)
        // We split by \n and process complete lines, keeping incomplete ones in buffer
        
        const lines = buffer.split('\n')
        buffer = lines.pop() || '' // Keep last incomplete line in buffer

        for (const line of lines) {
          // SSE events start with "data: "
          if (line.startsWith('data: ')) {
            try {
              // Parse JSON payload after "data: " prefix
              const event = JSON.parse(line.slice(6))

              // =============================================================================
              // Handle different SSE event types
              // =============================================================================
              
              if (event.type === 'status') {
                // Status update: "Planning...", "Researching...", "Generating...", "Evaluating..."
                // Show this in the UI so user knows what's happening
                setStreamingStatus(event.message)
                
              } else if (event.type === 'chunk') {
                // Execution output chunk - accumulate for potential use
                // NOTE: Currently chunks contain raw JSON from LLM, not clean text
                // We accumulate but don't display until we get the 'complete' event
                // which has the properly parsed final_output
                currentOutput += event.content
                
                // Update status to show we're actively generating
                setStreamingStatus('Generating...')
                
              } else if (event.type === 'complete') {
                // Stream completed - final result with evaluation
                setStreamingStatus(null) // Clear status
                setResult(event.final_output)
                setEvaluation(event.evaluation)
                setMemoryUsed(event.memory_used)

                // Refresh history if task passed with high score
                if (event.evaluation.passed && event.evaluation.score >= 8) {
                  setTimeout(async () => {
                    try {
                      const historyResponse = await fetch('http://localhost:8000/history')
                      if (historyResponse.ok) {
                        const historyData: HistoryResponse = await historyResponse.json()
                        setHistory(historyData.memories)
                      }
                    } catch (err) {
                      console.warn('Failed to refresh history:', err)
                    }
                  }, 1000)
                }
                
              } else if (event.type === 'error') {
                // Error event from backend
                setStreamingStatus(null)
                throw new Error(event.message)
              }
              
            } catch (parseError) {
              // Failed to parse SSE event JSON - log but continue
              console.warn('Failed to parse SSE event:', line, parseError)
            }
          }
        }
      }
      
    } finally {
      // Always clean up state when streaming ends (success or failure)
      setLoading(false)
      setStreamingStatus(null)
    }
  }

  const handleNonStreamingRequest = async (goalText: string) => {
    try {
      // =============================================================================
      // Call backend API without streaming (fallback mode)
      // =============================================================================
      // This is used when:
      // 1. Streaming is explicitly disabled
      // 2. Streaming failed and we're falling back
      // 3. Browser doesn't support ReadableStream
      
      const response = await fetch('http://localhost:8000/run', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ goal: goalText }),
      })

      // Parse response
      const data = await response.json()

      // Handle errors (4xx, 5xx)
      if (!response.ok) {
        const errorData = data.detail as ErrorResponse
        throw new Error(errorData?.message || 'Failed to execute task')
      }

      // =============================================================================
      // Success: Update state with complete response
      // =============================================================================
      // Unlike streaming, we receive everything at once:
      // - Final output (complete text)
      // - Evaluation results (passed, score, reasons)
      // - Memory usage flag
      
      const apiResponse = data as ApiResponse
      setResult(apiResponse.final_output)
      setEvaluation(apiResponse.evaluation)
      setMemoryUsed(apiResponse.memory_used)

      // =============================================================================
      // Refresh history after successful execution
      // =============================================================================
      // Only refresh if task passed with high quality (score >= 8)
      // This ensures history only shows successful, high-quality outputs
      
      if (apiResponse.evaluation.passed && apiResponse.evaluation.score >= 8) {
        setTimeout(async () => {
          try {
            const historyResponse = await fetch('http://localhost:8000/history')
            if (historyResponse.ok) {
              const historyData: HistoryResponse = await historyResponse.json()
              setHistory(historyData.memories)
            }
          } catch (err) {
            console.warn('Failed to refresh history:', err)
          }
        }, 1000)
      }
      
    } catch (err) {
      // Handle network errors, API errors, timeouts
      if (err instanceof Error) {
        setError(err.message)
      } else {
        setError('An unexpected error occurred. Please try again.')
      }
      console.error('Agent execution error:', err)
      
    } finally {
      setLoading(false)
    }
  }

  // =============================================================================
  // Clear Results Handler
  // =============================================================================

  const handleClear = () => {
    setGoal('')
    setResult(null)
    setEvaluation(null)
    setMemoryUsed(null)
    setError(null)
    setStreamingStatus(null)
  }

  return (
    <main className="container">
      {/* Header Section */}
      <header className="header">
        <div className="badge">Beta</div>
        <h1>AgentOps AI</h1>
        <p>
          Multi-agent task execution system
        </p>
      </header>

      {/* Main Card - Input Form */}
      <div className="card">
        <h2>Task Input</h2>
        <p className="text-muted" style={{ marginBottom: 'var(--space-lg)' }}>
          Enter a goal. The system plans, executes, and evaluates the output.
        </p>

        <AgentInput
          goal={goal}
          setGoal={setGoal}
          loading={loading}
          onSubmit={handleRunAgent}
          onClear={handleClear}
        />
      </div>

      {/* Streaming Status Indicator */}
      {loading && streamingStatus && (
        <div className="card" style={{ borderLeft: '4px solid var(--color-accent)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-md)' }}>
            <span className="spinner"></span>
            <div>
              <strong style={{ color: 'var(--color-accent)' }}>{streamingStatus}</strong>
              <p className="text-muted" style={{ marginTop: 'var(--space-xs)', marginBottom: 0, fontSize: '0.875rem' }}>
                {streamingStatus === 'Generating...' && 'Output is streaming in real-time below'}
                {streamingStatus === 'Planning task...' && 'Creating execution plan'}
                {streamingStatus === 'Conducting research...' && 'Gathering context and information'}
                {streamingStatus === 'Evaluating output...' && 'Checking quality and completeness'}
                {!['Generating...', 'Planning task...', 'Conducting research...', 'Evaluating output...'].includes(streamingStatus) && 'Processing your request'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Result Display */}
      {(result || error) && (
        <ResultDisplay
          result={result}
          evaluation={evaluation}
          memoryUsed={memoryUsed}
          error={error}
        />
      )}

      {/* Info Section - Only show when no results */}
      {!result && !error && (
        <div className="card">
          <h2>Agent Pipeline</h2>
          <div style={{ display: 'grid', gap: 'var(--space-lg)' }}>
            <div>
              <strong>Supervisor</strong>
              <p className="text-muted" style={{ marginTop: 'var(--space-xs)', marginBottom: 0 }}>
                Plans steps and determines if research is required.
              </p>
            </div>
            <div>
              <strong>Research (Conditional)</strong>
              <p className="text-muted" style={{ marginTop: 'var(--space-xs)', marginBottom: 0 }}>
                Gathers context and information when needed.
              </p>
            </div>
            <div>
              <strong>Execution</strong>
              <p className="text-muted" style={{ marginTop: 'var(--space-xs)', marginBottom: 0 }}>
                Generates output based on plan and research.
              </p>
            </div>
            <div>
              <strong>Evaluation</strong>
              <p className="text-muted" style={{ marginTop: 'var(--space-xs)', marginBottom: 0 }}>
                Validates output. Retries up to 5 times if needed.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Past Successful Tasks - Always visible */}
      <HistoryList history={history} loading={historyLoading} />

      {/* Footer */}
      <footer style={{ textAlign: 'center', color: 'var(--color-text-secondary)', marginTop: 'var(--space-2xl)', marginBottom: 'var(--space-xl)' }}>
        <p style={{ fontSize: '0.8125rem', margin: 0 }}>
          LangGraph • FastAPI • Next.js
        </p>
      </footer>
    </main>
  )
}
