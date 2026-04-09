import { useState, useRef } from 'react'
import { uploadCSV, runPipeline } from '../api/client.js'

export default function UploadPage({ onComplete }) {
  const [file, setFile] = useState(null)
  const [apiKey, setApiKey] = useState('')
  const [mode, setMode] = useState('rules')
  const [status, setStatus] = useState('')
  const [progress, setProgress] = useState([])
  const [running, setRunning] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const [error, setError] = useState('')
  const inputRef = useRef()

  const handleFile = (f) => {
    if (!f) return
    if (!f.name.toLowerCase().endsWith('.csv')) {
      setError('Please select a .csv file')
      return
    }
    setError('')
    setFile(f)
  }

  const handleRun = async () => {
    if (!file) { setError('Please select a CSV file first'); return }
    if (mode === 'llm' && !apiKey.trim()) { setError('Please enter your Gemini API key'); return }

    setRunning(true)
    setError('')
    setProgress([])
    setStatus('Uploading file…')

    try {
      const { data: upload } = await uploadCSV(file)
      setStatus('Pipeline running…')

      const results = await runPipeline(
        upload.session_id,
        apiKey.trim(),
        apiKey.trim() ? mode : 'rules',
        (event, data) => {
          if (event === 'started') setStatus('Pipeline started…')
          if (event === 'progress') {
            setProgress(prev => [...prev, data.label])
            setStatus(data.label)
          }
        }
      )

      onComplete(results)
    } catch (err) {
      setError(err.message || 'Something went wrong')
      setStatus('')
    } finally {
      setRunning(false)
    }
  }

  return (
    <div style={s.page}>
      <div style={s.card}>
        <h1 style={s.title}>🧹 Agentic Data Cleaner</h1>
        <p style={s.subtitle}>
          Upload a messy CSV. A multi-agent AI pipeline cleans it with per-column rationale.
        </p>

        {/* ── Drop zone ── */}
        <div
          style={{ ...s.dropzone, ...(dragOver ? s.dropzoneActive : {}), ...(file ? s.dropzoneFilled : {}) }}
          onClick={() => !running && inputRef.current.click()}
          onDragOver={(e) => { e.preventDefault(); if (!running) setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => { e.preventDefault(); setDragOver(false); if (!running) handleFile(e.dataTransfer.files[0]) }}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".csv"
            style={{ display: 'none' }}
            onChange={(e) => handleFile(e.target.files[0])}
          />
          {file ? (
            <>
              <div style={s.fileIcon}>📄</div>
              <div style={s.fileName}>{file.name}</div>
              <div style={s.fileSize}>{(file.size / 1024).toFixed(1)} KB — click to change</div>
            </>
          ) : (
            <>
              <div style={s.fileIcon}>📂</div>
              <div style={{ fontWeight: 500 }}>Drop a CSV here or click to browse</div>
              <div style={s.fileSize}>Max 50 MB</div>
            </>
          )}
        </div>

        {/* ── Mode selector ── */}
        <div style={s.row}>
          <label style={s.label}>Cleaning Mode</label>
          <select
            style={s.select}
            value={mode}
            onChange={e => { setMode(e.target.value); setError('') }}
            disabled={running}
          >
            <option value="rules">Rules — deterministic, no API key needed</option>
            <option value="llm">LLM — Gemini reasons each column</option>
          </select>
        </div>

        {/* ── Gemini API key (shown only in LLM mode) ── */}
        {mode === 'llm' && (
          <div style={s.row}>
            <label style={s.label}>
              Gemini API Key
              <a
                href="https://aistudio.google.com/app/apikey"
                target="_blank"
                rel="noreferrer"
                style={s.keyLink}
              >
                Get one free →
              </a>
            </label>
            <input
              style={s.input}
              type="password"
              placeholder="AIzaSy…"
              value={apiKey}
              onChange={e => { setApiKey(e.target.value); setError('') }}
              disabled={running}
              autoComplete="off"
            />
          </div>
        )}

        {/* ── Error banner ── */}
        {error && (
          <div style={s.errorBanner}>
            ⚠ {error}
          </div>
        )}

        {/* ── Run button ── */}
        <button
          style={{ ...s.btn, ...(running ? s.btnRunning : {}) }}
          onClick={handleRun}
          disabled={running}
        >
          {running ? (
            <span>⏳ {status || 'Running…'}</span>
          ) : (
            '▶  Clean Data'
          )}
        </button>

        {/* ── Progress list ── */}
        {progress.length > 0 && (
          <div style={s.progressBox}>
            {progress.map((label, i) => (
              <div key={i} style={s.progressItem}>
                <span style={s.check}>✓</span> {label}
              </div>
            ))}
            {running && (
              <div style={{ ...s.progressItem, color: '#2563eb' }}>
                <span style={s.spinner}>⏳</span> {status}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

const s = {
  page: {
    minHeight: '100vh',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'linear-gradient(135deg, #e0e7ff 0%, #f0f4f8 100%)',
    fontFamily: 'system-ui, -apple-system, sans-serif',
    padding: '24px 16px',
  },
  card: {
    background: '#fff',
    borderRadius: 16,
    padding: '40px 36px',
    maxWidth: 520,
    width: '100%',
    boxShadow: '0 8px 32px rgba(0,0,0,0.10)',
  },
  title: { margin: '0 0 8px', fontSize: 26, fontWeight: 800, color: '#0f172a' },
  subtitle: { color: '#64748b', marginBottom: 28, lineHeight: 1.6, fontSize: 15 },

  dropzone: {
    border: '2px dashed #cbd5e1',
    borderRadius: 10,
    padding: '32px 20px',
    textAlign: 'center',
    cursor: 'pointer',
    color: '#64748b',
    marginBottom: 22,
    transition: 'all .15s',
    fontSize: 14,
  },
  dropzoneActive: { borderColor: '#3b82f6', background: '#eff6ff', color: '#1d4ed8' },
  dropzoneFilled: { borderColor: '#22c55e', background: '#f0fdf4' },
  fileIcon: { fontSize: 34, marginBottom: 8 },
  fileName: { fontWeight: 700, color: '#1e293b', fontSize: 15 },
  fileSize: { fontSize: 12, color: '#94a3b8', marginTop: 4 },

  row: { marginBottom: 16 },
  label: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    fontSize: 13,
    fontWeight: 600,
    color: '#374151',
    marginBottom: 6,
  },
  keyLink: { fontSize: 12, color: '#2563eb', textDecoration: 'none', fontWeight: 400 },
  select: {
    width: '100%',
    padding: '9px 12px',
    borderRadius: 8,
    border: '1.5px solid #d1d5db',
    fontSize: 14,
    color: '#1e293b',
    background: '#fff',
    cursor: 'pointer',
  },
  input: {
    width: '100%',
    padding: '9px 12px',
    borderRadius: 8,
    border: '1.5px solid #d1d5db',
    fontSize: 14,
    color: '#1e293b',
    boxSizing: 'border-box',
    fontFamily: 'monospace',
  },

  errorBanner: {
    background: '#fef2f2',
    border: '1px solid #fecaca',
    color: '#dc2626',
    borderRadius: 8,
    padding: '10px 14px',
    fontSize: 13,
    marginBottom: 14,
  },

  btn: {
    width: '100%',
    padding: '13px',
    background: '#2563eb',
    color: '#fff',
    border: 'none',
    borderRadius: 10,
    fontSize: 16,
    fontWeight: 700,
    cursor: 'pointer',
    marginTop: 4,
    transition: 'background .15s',
    letterSpacing: '.3px',
  },
  btnRunning: { background: '#93c5fd', cursor: 'not-allowed' },

  progressBox: {
    marginTop: 20,
    background: '#f8fafc',
    border: '1px solid #e2e8f0',
    borderRadius: 10,
    padding: '14px 16px',
  },
  progressItem: { padding: '4px 0', fontSize: 14, color: '#334155', display: 'flex', gap: 8 },
  check: { color: '#22c55e', fontWeight: 700 },
  spinner: { animation: 'spin 1s linear infinite' },
}