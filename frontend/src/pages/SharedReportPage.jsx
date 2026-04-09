import React, { useState, useEffect } from 'react'
import ResultsPage from './ResultsPage'

export default function SharedReportPage({ runId }) {
  const [state, setState] = useState({ status: 'loading', results: null, error: '' })

  useEffect(() => {
    if (!runId) { setState({ status: 'error', error: 'No run ID in URL.' }); return }
    fetch(`/api/results/${runId}`)
      .then(r => { if (!r.ok) throw new Error('Run not found'); return r.json() })
      .then(data => setState({
        status: 'done',
        results: {
          ...data,
          csv_download_url: `/api/download/${runId}/csv`,
          report_download_url: `/api/download/${runId}/report`,
        }
      }))
      .catch(e => setState({ status: 'error', error: e.message }))
  }, [runId])

  if (state.status === 'loading') return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '100vh', fontFamily: 'system-ui, sans-serif', color: '#6b7280' }}>
      Loading report…
    </div>
  )

  if (state.status === 'error') return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '100vh', fontFamily: 'system-ui, sans-serif' }}>
      <div style={{ textAlign: 'center' }}>
        <div style={{ fontSize: 40, marginBottom: 16 }}>⚠</div>
        <p style={{ fontSize: 15, color: '#dc2626', marginBottom: 8 }}>{state.error}</p>
        <p style={{ fontSize: 13, color: '#9a9a94' }}>This report may have expired or the run ID is invalid.</p>
        <a href='/' style={{ marginTop: 16, display: 'inline-block', padding: '10px 24px', borderRadius: 8, background: '#1a1a18', color: '#fff', textDecoration: 'none', fontSize: 13 }}>Go to app</a>
      </div>
    </div>
  )

  return (
    <div style={{ minHeight: '100vh', background: '#f5f4f0', fontFamily: 'system-ui, sans-serif' }}>
      <div style={{ background: '#fff', borderBottom: '0.5px solid #e5e3db', padding: '10px 24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{ fontSize: 13, fontWeight: 600, color: '#1a1a18' }}>🧹 Agentic Data Cleaner — Shared Report</span>
        <a href='/' style={{ fontSize: 12, color: '#6b7280', textDecoration: 'none' }}>Open app →</a>
      </div>
      <ResultsPage results={state.results} onReset={() => window.location.href = '/'} apiKey='' />
    </div>
  )
}