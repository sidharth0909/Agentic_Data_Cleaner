import React, { useState, useEffect } from 'react'
import UploadPage from './pages/UploadPage'
import ResultsPage from './pages/ResultsPage'
import HistoryPage, { saveRunToHistory } from './pages/HistoryPage'
import SharedReportPage from './pages/SharedReportPage'

function getSharedRunId() {
  const p = window.location.pathname
  const m = p.match(/^\/report\/([a-zA-Z0-9_-]+)$/)
  return m ? m[1] : null
}

export default function App() {
  const [page, setPage]     = useState('upload') // upload | results | history
  const [results, setResults] = useState(null)
  const sharedRunId = getSharedRunId()

  if (sharedRunId) return <SharedReportPage runId={sharedRunId} />

  const handleComplete = (res) => {
    saveRunToHistory(res)
    setResults(res)
    setPage('results')
  }

  return (
    <div style={{ minHeight: '100vh', background: '#f5f4f0', fontFamily: 'system-ui, sans-serif' }}>
      {page === 'upload' && (
        <UploadPage
          onComplete={handleComplete}
          onHistory={() => setPage('history')}
        />
      )}
      {page === 'results' && (
        <ResultsPage
          results={results}
          onReset={() => { setResults(null); setPage('upload') }}
          apiKey={results?._api_key || ''}
        />
      )}
      {page === 'history' && (
        <HistoryPage
          onOpen={(res) => { setResults(res); setPage('results') }}
          onBack={() => setPage('upload')}
        />
      )}
    </div>
  )
}