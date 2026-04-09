import axios from 'axios'

const BASE = import.meta.env.VITE_API_BASE_URL || ''

export const uploadCSV = (file) => {
  const form = new FormData()
  form.append('file', file)
  return axios.post(`${BASE}/api/upload`, form)
}

export const downloadFile = (runId, type) =>
  `${BASE}/api/download/${runId}/${type}`

/**
 * Run the pipeline and stream SSE events via fetch + ReadableStream.
 * Each SSE message is processed exactly once as it arrives.
 *
 * onEvent(eventName, data) is called for every event.
 * Resolves with the "done" payload, or rejects with an Error.
 */
export const runPipeline = (sessionId, apiKey, mode, overrides, onEvent) => {
  return new Promise(async (resolve, reject) => {
    let response
    try {
      response = await fetch(`${BASE}/api/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          api_key: apiKey,
          mode,
          max_iterations: 3,
          quality_threshold: 0.90,
          column_overrides: overrides || {},
        }),
      })
    } catch (err) {
      // Network-level failure — backend probably not running
      return reject(new Error(
        'Cannot reach the backend. Make sure the server is running on port 8000.\n' +
        '  cd backend && uvicorn main:app --reload'
      ))
    }

    if (!response.ok) {
      let detail = `Server returned ${response.status}`
      try { const j = await response.json(); detail = j.detail || detail } catch {}
      return reject(new Error(detail))
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    const processBuffer = () => {
      // SSE messages are separated by double newlines
      const messages = buffer.split('\n\n')
      // Keep the last (possibly incomplete) chunk in the buffer
      buffer = messages.pop()

      for (const message of messages) {
        if (!message.trim()) continue

        let eventName = 'message'
        let dataStr = ''

        for (const line of message.split('\n')) {
          if (line.startsWith('event:')) {
            eventName = line.slice(6).trim()
          } else if (line.startsWith('data:')) {
            dataStr = line.slice(5).trim()
          }
        }

        if (!dataStr) continue

        let data
        try {
          data = JSON.parse(dataStr)
        } catch {
          continue
        }

        onEvent(eventName, data)

        if (eventName === 'done') { resolve(data); return true }
        if (eventName === 'error') { reject(new Error(data.detail || 'Pipeline error')); return true }
      }
      return false
    }

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        if (processBuffer()) break
      }
    } catch (err) {
      reject(new Error('Stream error: ' + err.message))
    }
  })
}