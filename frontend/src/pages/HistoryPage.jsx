import React, { useState, useEffect } from 'react'

const STORAGE_KEY = 'adc_run_history'

export function saveRunToHistory(results) {
  try {
    const existing = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')
    const entry = {
      run_id: results.run_id,
      filename: results.filename,
      score: results.final_score?.overall ?? 0,
      rows_before: results.shape_before?.[0] ?? 0,
      cols_before: results.shape_before?.[1] ?? 0,
      rows_after: results.shape_after?.[0] ?? 0,
      cols_after: results.shape_after?.[1] ?? 0,
      iterations: results.iterations_completed ?? 1,
      timestamp: Date.now(),
    }
    const updated = [entry, ...existing.filter(e => e.run_id !== entry.run_id)].slice(0, 20)
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated))
  } catch {}
}

export default function HistoryPage({ onOpen, onBack }) {
  const [runs, setRuns] = useState([])
  const [loading, setLoading] = useState(null)

  useEffect(() => {
    try {
      setRuns(JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]'))
    } catch { setRuns([]) }
  }, [])

  const clear = () => { localStorage.removeItem(STORAGE_KEY); setRuns([]) }

  const open = async (run) => {
    setLoading(run.run_id)
    try {
      const res = await fetch(`/api/results/${run.run_id}`)
      if (!res.ok) throw new Error('Run not found on server')
      const data = await res.json()
      onOpen({ ...data, csv_download_url: `/api/download/${run.run_id}/csv`, report_download_url: `/api/download/${run.run_id}/report` })
    } catch (e) {
      alert(`Could not load run: ${e.message}`)
    } finally { setLoading(null) }
  }

  const scoreColor = s => s >= 0.9 ? '#059669' : s >= 0.7 ? '#d97706' : '#dc2626'
  const scoreBg    = s => s >= 0.9 ? '#d1fae5' : s >= 0.7 ? '#fef3c7' : '#fee2e2'

  return (
    <div style={{ maxWidth: 860, margin: '0 auto', padding: '3rem 1.5rem', fontFamily: 'system-ui, sans-serif' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '2rem' }}>
        <div>
          <p style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.12em', textTransform: 'uppercase', color: '#9a9a94', marginBottom: 4 }}>Run history</p>
          <h1 style={{ fontSize: 22, fontWeight: 600, color: '#1a1a18' }}>Past cleaning runs</h1>
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          {runs.length > 0 && <button onClick={clear} style={{ padding: '8px 14px', borderRadius: 8, fontSize: 13, border: '0.5px solid #fecaca', background: '#fff', color: '#dc2626', cursor: 'pointer' }}>Clear history</button>}
          <button onClick={onBack} style={{ padding: '8px 16px', borderRadius: 8, fontSize: 13, border: '0.5px solid #d3d1c7', background: '#fff', cursor: 'pointer', color: '#5a5a56' }}>← New run</button>
        </div>
      </div>

      {runs.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '4rem 2rem', color: '#9a9a94' }}>
          <div style={{ fontSize: 40, marginBottom: 16 }}>📂</div>
          <p style={{ fontSize: 15, marginBottom: 8 }}>No runs yet</p>
          <p style={{ fontSize: 13 }}>Clean a CSV and your runs will appear here.</p>
          <button onClick={onBack} style={{ marginTop: 20, padding: '10px 24px', borderRadius: 8, background: '#1a1a18', color: '#fff', border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 600 }}>Clean a file ▶</button>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {runs.map(run => (
            <div key={run.run_id} style={{ background: '#fff', border: '0.5px solid #e5e3db', borderRadius: 12, padding: '14px 18px', display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
              <div style={{ flex: 1, minWidth: 160 }}>
                <div style={{ fontWeight: 600, fontSize: 14, color: '#1a1a18', marginBottom: 3 }}>{run.filename}</div>
                <div style={{ fontSize: 11, color: '#9a9a94' }}>{new Date(run.timestamp).toLocaleString()}</div>
              </div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
                <span style={{ fontSize: 13, fontWeight: 700, padding: '3px 10px', borderRadius: 99, background: scoreBg(run.score), color: scoreColor(run.score) }}>{Math.round(run.score * 100)}%</span>
                <span style={{ fontSize: 11, color: '#6b7280', background: '#f3f4f6', padding: '3px 8px', borderRadius: 99 }}>{run.rows_before.toLocaleString()} → {run.rows_after.toLocaleString()} rows</span>
                <span style={{ fontSize: 11, color: '#6b7280', background: '#f3f4f6', padding: '3px 8px', borderRadius: 99 }}>{run.cols_before} → {run.cols_after} cols</span>
                <span style={{ fontSize: 11, color: '#6b7280', background: '#f3f4f6', padding: '3px 8px', borderRadius: 99 }}>{run.iterations} iter</span>
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <a href={`/api/download/${run.run_id}/csv`} download style={{ padding: '6px 12px', borderRadius: 7, fontSize: 12, border: '0.5px solid #d3d1c7', background: '#fff', color: '#1a1a18', textDecoration: 'none' }}>↓ CSV</a>
                <a href={`/api/download/${run.run_id}/parquet`} download style={{ padding: '6px 12px', borderRadius: 7, fontSize: 12, border: '0.5px solid #d3d1c7', background: '#fff', color: '#1a1a18', textDecoration: 'none' }}>↓ Parquet</a>
                <button onClick={() => open(run)} disabled={loading === run.run_id} style={{ padding: '6px 14px', borderRadius: 7, fontSize: 12, border: 'none', background: '#1a1a18', color: '#fff', cursor: 'pointer', fontWeight: 600 }}>
                  {loading === run.run_id ? '…' : 'View results'}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}