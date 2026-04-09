import React, { useState } from 'react'
import UploadPage from './pages/UploadPage'
import ResultsPage from './pages/ResultsPage'

export default function App() {
  const [results, setResults] = useState(null)

  return (
    <div style={{ minHeight: '100vh', background: '#f5f4f0', fontFamily: 'system-ui, sans-serif' }}>
      {results
        ? <ResultsPage results={results} onReset={() => setResults(null)} />
        : <UploadPage onResults={setResults} />
      }
    </div>
  )
}
