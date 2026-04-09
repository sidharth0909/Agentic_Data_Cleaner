import React, { useState, useEffect, useRef } from 'react'

// ─── Action metadata ──────────────────────────────────────────────────────────
const ACTION_META = {
  impute_mean:     { color: '#2563eb', bg: '#dbeafe', label: 'impute mean' },
  impute_median:   { color: '#2563eb', bg: '#dbeafe', label: 'impute median' },
  impute_mode:     { color: '#2563eb', bg: '#dbeafe', label: 'impute mode' },
  impute_constant: { color: '#2563eb', bg: '#dbeafe', label: 'impute const' },
  drop_column:     { color: '#dc2626', bg: '#fee2e2', label: 'drop column' },
  clip_iqr:        { color: '#d97706', bg: '#fef3c7', label: 'clip IQR' },
  winsorise:       { color: '#d97706', bg: '#fef3c7', label: 'winsorise' },
  encode_onehot:   { color: '#7c3aed', bg: '#ede9fe', label: 'one-hot encode' },
  encode_ordinal:  { color: '#7c3aed', bg: '#ede9fe', label: 'ordinal encode' },
  encode_binary:   { color: '#7c3aed', bg: '#ede9fe', label: 'binary encode' },
  scale_standard:  { color: '#059669', bg: '#d1fae5', label: 'standardise' },
  scale_minmax:    { color: '#059669', bg: '#d1fae5', label: 'min-max scale' },
  scale_robust:    { color: '#059669', bg: '#d1fae5', label: 'robust scale' },
  log_transform:   { color: '#0891b2', bg: '#cffafe', label: 'log transform' },
  sqrt_transform:  { color: '#0891b2', bg: '#cffafe', label: 'sqrt transform' },
  keep:            { color: '#6b7280', bg: '#f3f4f6', label: 'keep as-is' },
}

// === PASTE THIS BLOCK into ResultsPage.jsx right after the ACTION_META const ===
// It adds: HistogramChart, ValueCountsBar, AlertsPanel, AISummaryCard components
// Plus the new tabs in the TABS array and their render sections

// ─── SVG Histogram (before vs after overlay) ────────────────────────────────
function HistogramChart({ before, after, column, action }) {
  if (!before?.counts || !before?.edges) return null
  const W = 260, H = 90, PAD = { l: 6, r: 6, t: 10, b: 18 }
  const bW = W - PAD.l - PAD.r
  const bH = H - PAD.t - PAD.b
  const n = before.counts.length
  const maxCount = Math.max(...before.counts, ...(after?.counts || [0]))
  if (maxCount === 0) return null
  const barW = bW / n

  const bar = (counts, color, opacity) =>
    counts.map((c, i) => {
      const h = (c / maxCount) * bH
      return (
        <rect key={i}
          x={PAD.l + i * barW} y={PAD.t + bH - h}
          width={barW - 1} height={h}
          fill={color} opacity={opacity} rx={1}
        />
      )
    })

  const edgeFmt = v => Math.abs(v) >= 1000 ? `${(v/1000).toFixed(1)}k`
                    : Math.abs(v) >= 1 ? v.toFixed(1)
                    : v.toFixed(2)

  return (
    <svg width={W} height={H} style={{ display: 'block', overflow: 'visible' }}>
      {/* Before bars (grey) */}
      {bar(before.counts, '#94a3b8', 0.6)}
      {/* After bars (colored) */}
      {after?.counts && bar(after.counts, '#2563eb', 0.75)}
      {/* X axis */}
      <line x1={PAD.l} y1={PAD.t + bH} x2={PAD.l + bW} y2={PAD.t + bH} stroke='#e2e8f0' strokeWidth={0.5}/>
      {/* Min / max labels */}
      <text x={PAD.l} y={H - 2} fontSize={8} fill='#94a3b8'>{edgeFmt(before.edges[0])}</text>
      <text x={PAD.l + bW} y={H - 2} fontSize={8} fill='#94a3b8' textAnchor='end'>{edgeFmt(before.edges[before.edges.length - 1])}</text>
    </svg>
  )
}

// ─── Value counts bar (categorical) ─────────────────────────────────────────
function ValueCountsBar({ valueCounts }) {
  if (!valueCounts?.length) return null
  const top = valueCounts.slice(0, 6)
  const maxPct = top[0]?.pct ?? 1
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4, width: 260 }}>
      {top.map((vc, i) => (
        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ fontSize: 10, color: '#6b7280', width: 80, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flexShrink: 0 }}>{vc.value}</span>
          <div style={{ flex: 1, background: '#f1f5f9', borderRadius: 2, height: 10, position: 'relative', overflow: 'hidden' }}>
            <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: `${(vc.pct / maxPct) * 100}%`, background: '#7c3aed', opacity: 0.7, borderRadius: 2 }} />
          </div>
          <span style={{ fontSize: 10, color: '#6b7280', width: 32, textAlign: 'right', flexShrink: 0 }}>{(vc.pct * 100).toFixed(0)}%</span>
        </div>
      ))}
    </div>
  )
}

// ─── Alerts Panel ────────────────────────────────────────────────────────────
function AlertsPanel({ profiles, cleaningPlan }) {
  if (!profiles?.length) return null

  const planByCol = {}
  ;(cleaningPlan || []).forEach(s => { planByCol[s.column] = s })

  const alerts = []

  profiles.forEach(p => {
    const step = planByCol[p.column]
    const actionLabel = step ? step.action.replace(/_/g, ' ') : null

    if ((p.missing_pct ?? 0) > 0.5)
      alerts.push({ level: 'high', col: p.column, msg: `${(p.missing_pct*100).toFixed(0)}% missing values`, fix: actionLabel, rationale: step?.rationale })
    else if ((p.missing_pct ?? 0) > 0.05)
      alerts.push({ level: 'med', col: p.column, msg: `${(p.missing_pct*100).toFixed(1)}% missing values`, fix: actionLabel, rationale: step?.rationale })

    if ((p.outlier_ratio ?? 0) > 0.1)
      alerts.push({ level: 'high', col: p.column, msg: `${(p.outlier_ratio*100).toFixed(1)}% outliers (IQR)`, fix: actionLabel, rationale: step?.rationale })
    else if ((p.outlier_ratio ?? 0) > 0.05)
      alerts.push({ level: 'med', col: p.column, msg: `${(p.outlier_ratio*100).toFixed(1)}% outliers`, fix: actionLabel, rationale: step?.rationale })

    if (p.skewness != null && Math.abs(p.skewness) > 2)
      alerts.push({ level: 'high', col: p.column, msg: `Highly skewed (${p.skewness.toFixed(2)})`, fix: actionLabel, rationale: step?.rationale })
    else if (p.skewness != null && Math.abs(p.skewness) > 1)
      alerts.push({ level: 'med', col: p.column, msg: `Skewed distribution (${p.skewness.toFixed(2)})`, fix: actionLabel, rationale: step?.rationale })

    if (p.cardinality_hint === 'identifier' && step?.action !== 'keep' && step?.action !== 'drop_column')
      alerts.push({ level: 'info', col: p.column, msg: 'Possible ID column — high unique ratio', fix: actionLabel, rationale: step?.rationale })

    if (p.missing_pct === 1)
      alerts.push({ level: 'high', col: p.column, msg: '100% null — no data', fix: actionLabel, rationale: step?.rationale })
  })

  // Sort: high → med → info
  const order = { high: 0, med: 1, info: 2 }
  alerts.sort((a, b) => order[a.level] - order[b.level])

  if (!alerts.length) return (
    <div style={{ background: '#f0fdf4', border: '0.5px solid #bbf7d0', borderRadius: 10, padding: '14px 18px', fontSize: 13, color: '#059669', display: 'flex', alignItems: 'center', gap: 10 }}>
      <span style={{ fontSize: 18 }}>✓</span>
      <span>No data quality alerts — dataset looks clean!</span>
    </div>
  )

  const LEVEL = {
    high: { bg: '#fef2f2', border: '#fecaca', dot: '#dc2626', text: '#991b1b', badge: 'High' },
    med:  { bg: '#fffbeb', border: '#fde68a', dot: '#d97706', text: '#92400e', badge: 'Warning' },
    info: { bg: '#eff6ff', border: '#bfdbfe', dot: '#2563eb', text: '#1e40af', badge: 'Info' },
  }

  const counts = { high: alerts.filter(a => a.level === 'high').length, med: alerts.filter(a => a.level === 'med').length }

  return (
    <div>
      {/* Summary bar */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 16, flexWrap: 'wrap' }}>
        {counts.high > 0 && <span style={{ fontSize: 12, fontWeight: 600, padding: '4px 12px', borderRadius: 99, background: '#fef2f2', color: '#dc2626', border: '0.5px solid #fecaca' }}>{counts.high} critical</span>}
        {counts.med > 0  && <span style={{ fontSize: 12, fontWeight: 600, padding: '4px 12px', borderRadius: 99, background: '#fffbeb', color: '#d97706', border: '0.5px solid #fde68a'  }}>{counts.med} warnings</span>}
        <span style={{ fontSize: 12, color: '#9a9a94', alignSelf: 'center' }}>All issues below were detected before cleaning — agent actions shown in green.</span>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {alerts.map((a, i) => {
          const L = LEVEL[a.level]
          return (
            <div key={i} style={{ background: L.bg, border: `0.5px solid ${L.border}`, borderRadius: 10, padding: '12px 14px' }}>
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
                <div style={{ width: 8, height: 8, borderRadius: '50%', background: L.dot, flexShrink: 0, marginTop: 4 }} />
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginBottom: a.fix ? 4 : 0 }}>
                    <span style={{ fontSize: 13, fontWeight: 600, color: '#1a1a18' }}>{a.col}</span>
                    <span style={{ fontSize: 12, color: '#4b5563' }}>—</span>
                    <span style={{ fontSize: 12, color: L.text }}>{a.msg}</span>
                  </div>
                  {a.fix && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span style={{ fontSize: 11, padding: '1px 8px', borderRadius: 99, background: '#dcfce7', color: '#15803d', fontWeight: 600 }}>
                        ✓ fixed: {a.fix}
                      </span>
                      {a.rationale && <span style={{ fontSize: 11, color: '#6b7280', lineHeight: 1.4 }}>{a.rationale.slice(0, 80)}{a.rationale.length > 80 ? '…' : ''}</span>}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ─── AI Summary Card ─────────────────────────────────────────────────────────
function AISummaryCard({ runId, apiKey, results }) {
  const [state, setState] = React.useState({ status: 'idle', text: '', source: '' })

  const generate = async () => {
    setState({ status: 'loading', text: '', source: '' })
    try {
      const res = await fetch(`/api/summary/${runId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key: apiKey || '' }),
      })
      const data = await res.json()
      setState({ status: 'done', text: data.summary, source: data.source })
    } catch (e) {
      setState({ status: 'error', text: 'Failed to generate summary.', source: '' })
    }
  }

  return (
    <div style={{ background: '#fff', border: '0.5px solid #e5e3db', borderRadius: 12, padding: '18px 20px', marginBottom: 20 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <div>
          <p style={{ fontSize: 13, fontWeight: 600, color: '#1a1a18', margin: 0 }}>AI Dataset Summary</p>
          <p style={{ fontSize: 11, color: '#9a9a94', margin: '2px 0 0' }}>Plain-English narrative of what was found and fixed</p>
        </div>
        {state.status !== 'loading' && (
          <button onClick={generate} style={{ padding: '6px 14px', borderRadius: 7, border: '0.5px solid #d3d1c7', background: state.status === 'done' ? '#f9f9f8' : '#1a1a18', color: state.status === 'done' ? '#6b7280' : '#fff', fontSize: 12, cursor: 'pointer', fontWeight: 500 }}>
            {state.status === 'done' ? '↺ Regenerate' : '✦ Generate'}
          </button>
        )}
      </div>

      {state.status === 'idle' && (
        <p style={{ fontSize: 13, color: '#9a9a94', fontStyle: 'italic' }}>
          Click Generate to create a plain-English summary of this dataset and cleaning run.
          {apiKey ? ' Using Gemini for AI-powered summary.' : ' No API key — will use rule-based summary.'}
        </p>
      )}
      {state.status === 'loading' && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, color: '#6b7280', fontSize: 13 }}>
          <span style={{ animation: 'spin 1s linear infinite', display: 'inline-block' }}>⏳</span> Generating summary…
        </div>
      )}
      {state.status === 'done' && (
        <div>
          <p style={{ fontSize: 14, color: '#1a1a18', lineHeight: 1.7, margin: 0 }}>{state.text}</p>
          <div style={{ marginTop: 10, display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: 10, padding: '1px 7px', borderRadius: 99, background: state.source === 'llm' ? '#dbeafe' : '#f3f4f6', color: state.source === 'llm' ? '#1d4ed8' : '#6b7280', fontWeight: 600 }}>
              {state.source === 'llm' ? '✦ Gemini' : 'rule-based'}
            </span>
          </div>
        </div>
      )}
      {state.status === 'error' && (
        <p style={{ fontSize: 13, color: '#dc2626' }}>{state.text}</p>
      )}
    </div>
  )
}


// ─── CSV parser ───────────────────────────────────────────────────────────────
function parseCSV(text) {
  const lines = text.trim().split('\n')
  if (!lines.length) return { headers: [], rows: [] }
  const splitRow = line => {
    const cells = []
    let cur = '', inQ = false
    for (let i = 0; i < line.length; i++) {
      const ch = line[i]
      if (ch === '"') { inQ = !inQ }
      else if (ch === ',' && !inQ) { cells.push(cur.trim()); cur = '' }
      else cur += ch
    }
    cells.push(cur.trim())
    return cells
  }
  const headers = splitRow(lines[0])
  const rows = lines.slice(1).map(splitRow)
  return { headers, rows }
}

// ─── Export CSV for selected columns ─────────────────────────────────────────
function exportCSV(headers, rows, selectedCols) {
  const colIdxs = headers.map((_, i) => i).filter(i => selectedCols.has(i))
  const colNames = colIdxs.map(i => headers[i])
  const escape = v => (v.includes(',') || v.includes('"') || v.includes('\n'))
    ? `"${v.replace(/"/g, '""')}"` : v
  const lines = [
    colNames.map(escape).join(','),
    ...rows.map(row => colIdxs.map(i => escape(row[i] ?? '')).join(','))
  ]
  const blob = new Blob([lines.join('\n')], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = 'cleaned_selected.csv'; a.click()
  URL.revokeObjectURL(url)
}


// ─── ML Readiness ─────────────────────────────────────────────────────────────
function mlReadiness(profile, step) {
  const issues = []
  const action = step?.action ?? 'keep'
  if ((profile.missing_pct ?? 0) > 0.3) issues.push('high missing')
  if ((profile.outlier_ratio ?? 0) > 0.1) issues.push('outliers')
  if (profile.skewness != null && Math.abs(profile.skewness) > 2) issues.push('skewed')
  if (profile.cardinality_hint === 'identifier') issues.push('ID column')
  if (profile.cardinality_hint === 'high' && profile.skewness == null) issues.push('high cardinality')
  if (action === 'drop_column') return { level: 'dropped', label: 'Dropped', color: '#dc2626', bg: '#fee2e2', tip: 'Column was removed' }
  if (issues.length === 0) return { level: 'ready', label: 'ML Ready', color: '#059669', bg: '#d1fae5', tip: 'No issues detected' }
  if (issues.length === 1) return { level: 'warn', label: 'Review', color: '#d97706', bg: '#fef3c7', tip: issues[0] }
  return { level: 'risk', label: 'High Risk', color: '#dc2626', bg: '#fee2e2', tip: issues.join(', ') }
}

// ─── Correlation Heatmap ──────────────────────────────────────────────────────
function CorrelationHeatmap({ correlation }) {
  const [hovered, setHovered] = React.useState(null)
  if (!correlation?.columns?.length) return (
    <div style={{padding:'2rem',textAlign:'center',color:'var(--color-text-secondary)',fontSize:13}}>No numeric columns available for correlation.</div>
  )
  const { columns, matrix, top_pairs } = correlation
  const n = columns.length
  const CELL = Math.min(42, Math.floor(460 / n))
  const LABEL_W = 90

  const colorForR = r => {
    if (r == null) return '#f1f5f9'
    const abs = Math.abs(r)
    if (r > 0) return `rgba(37,99,235,${0.1 + abs * 0.8})`
    return `rgba(220,38,38,${0.1 + abs * 0.8})`
  }
  const textForR = r => {
    if (r == null) return ''
    return Math.abs(r) > 0.3 ? r.toFixed(2) : ''
  }

  return (
    <div>
      <div style={{overflowX:'auto',marginBottom:20}}>
        <div style={{display:'inline-block',position:'relative'}}>
          {/* Column labels top */}
          <div style={{display:'flex',marginLeft:LABEL_W,marginBottom:2}}>
            {columns.map((c,i) => (
              <div key={i} style={{width:CELL,fontSize:9,color:'#6b7280',textAlign:'center',overflow:'hidden',transform:'rotate(-40deg)',transformOrigin:'bottom left',height:50,whiteSpace:'nowrap',marginLeft:2}}>{c}</div>
            ))}
          </div>
          {/* Rows */}
          {matrix.map((row, ri) => (
            <div key={ri} style={{display:'flex',alignItems:'center',marginBottom:2}}>
              <div style={{width:LABEL_W,fontSize:10,color:'#6b7280',textAlign:'right',paddingRight:8,overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap',flexShrink:0}}>{columns[ri]}</div>
              {row.map((val, ci) => (
                <div key={ci}
                  onMouseEnter={() => setHovered({r:ri,c:ci,val})}
                  onMouseLeave={() => setHovered(null)}
                  style={{
                    width:CELL, height:CELL, marginRight:2,
                    background: colorForR(val),
                    borderRadius:3, display:'flex', alignItems:'center', justifyContent:'center',
                    fontSize:8, fontWeight:600, color: val != null && Math.abs(val) > 0.5 ? '#fff' : '#374151',
                    cursor:'default', border: hovered?.r===ri && hovered?.c===ci ? '1.5px solid #1a1a18' : '1.5px solid transparent',
                    transition:'border 0.1s',
                  }}>
                  {ri === ci ? <div style={{width:6,height:6,borderRadius:'50%',background:'#cbd5e1'}}/> : textForR(val)}
                </div>
              ))}
            </div>
          ))}
        </div>
        {hovered && hovered.r !== hovered.c && (
          <div style={{marginTop:8,fontSize:12,color:'#374151',background:'#f8fafc',border:'0.5px solid #e2e8f0',borderRadius:6,padding:'6px 10px',display:'inline-block'}}>
            <strong>{columns[hovered.r]}</strong> × <strong>{columns[hovered.c]}</strong>: r = {hovered.val?.toFixed(4) ?? '—'}
          </div>
        )}
      </div>

      {/* Legend */}
      <div style={{display:'flex',alignItems:'center',gap:12,marginBottom:20,fontSize:11,color:'#6b7280'}}>
        <div style={{display:'flex',alignItems:'center',gap:4}}>
          <div style={{width:12,height:12,borderRadius:2,background:'rgba(220,38,38,0.8)'}}/>Negative
        </div>
        <div style={{display:'flex',alignItems:'center',gap:4}}>
          <div style={{width:12,height:12,borderRadius:2,background:'rgba(37,99,235,0.8)'}}/>Positive
        </div>
        <div style={{display:'flex',alignItems:'center',gap:4}}>
          <div style={{width:12,height:12,borderRadius:2,background:'#f1f5f9',border:'0.5px solid #e2e8f0'}}/>No data
        </div>
      </div>

      {/* Top pairs */}
      {top_pairs?.length > 0 && (
        <div>
          <p style={{fontSize:12,fontWeight:600,color:'#6b7280',textTransform:'uppercase',letterSpacing:'0.08em',marginBottom:10}}>Strongest correlations</p>
          <div style={{display:'flex',flexDirection:'column',gap:6}}>
            {top_pairs.slice(0,6).map((p,i) => {
              const abs = Math.abs(p.r)
              const strength = abs > 0.7 ? 'Strong' : abs > 0.4 ? 'Moderate' : 'Weak'
              const color = abs > 0.7 ? '#dc2626' : abs > 0.4 ? '#d97706' : '#6b7280'
              const bg = abs > 0.7 ? '#fee2e2' : abs > 0.4 ? '#fef3c7' : '#f3f4f6'
              return (
                <div key={i} style={{display:'flex',alignItems:'center',gap:10,padding:'8px 12px',background:'#fff',border:'0.5px solid #e5e3db',borderRadius:8}}>
                  <span style={{fontSize:11,fontWeight:600,padding:'1px 7px',borderRadius:99,background:bg,color,flexShrink:0}}>{strength}</span>
                  <span style={{fontSize:13,fontWeight:500,color:'#1a1a18'}}>{p.col_a}</span>
                  <span style={{fontSize:12,color:'#9a9a94'}}>×</span>
                  <span style={{fontSize:13,fontWeight:500,color:'#1a1a18'}}>{p.col_b}</span>
                  <span style={{fontSize:13,fontWeight:600,color: p.r > 0 ? '#2563eb' : '#dc2626',marginLeft:'auto'}}>{p.r > 0 ? '+' : ''}{p.r.toFixed(3)}</span>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

// ─── Mini bar ─────────────────────────────────────────────────────────────────
function MiniBar({ value, max = 1, color = '#2563eb', width = 120, height = 6 }) {
  const filled = Math.max(0, Math.min(1, value / max)) * width
  return (
    <svg width={width} height={height} style={{ borderRadius: 99, overflow: 'hidden', display: 'block' }}>
      <rect x={0} y={0} width={width} height={height} fill='#f1f5f9' rx={3} />
      <rect x={0} y={0} width={filled} height={height} fill={color} rx={3} />
    </svg>
  )
}

// ─── Skewness gauge ───────────────────────────────────────────────────────────
function SkewnessGauge({ skewness }) {
  if (skewness == null) return <span style={{ fontSize: 12, color: '#9a9a94' }}>n/a</span>
  const clamped = Math.max(-3, Math.min(3, skewness))
  const t = (clamped + 3) / 6
  const angle = Math.PI - t * Math.PI
  const cx = 28, cy = 26, r = 20
  const x = cx + r * Math.cos(angle)
  const y = cy - r * Math.sin(angle)
  const needleColor = Math.abs(skewness) < 0.5 ? '#059669' : Math.abs(skewness) < 1.5 ? '#d97706' : '#dc2626'
  return (
    <svg width={56} height={30} style={{ display: 'block' }}>
      <path d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`} fill='none' stroke='#e2e8f0' strokeWidth={4} strokeLinecap='round' />
      <line x1={cx} y1={cy} x2={x} y2={y} stroke={needleColor} strokeWidth={2.5} strokeLinecap='round' />
      <circle cx={cx} cy={cy} r={3} fill={needleColor} />
      <text x={cx} y={cy + 10} textAnchor='middle' fontSize={8} fill='#9a9a94'>{skewness.toFixed(2)}</text>
    </svg>
  )
}

// ─── Column card ──────────────────────────────────────────────────────────────
function ColumnCard({ profile, step }) {
  const action = step?.action ?? 'keep'
  const meta = ACTION_META[action] ?? ACTION_META.keep
  const missing = profile.missing_pct ?? 0
  const outlier = profile.outlier_ratio ?? 0
  const isNumeric = profile.skewness != null
  const missingColor = missing > 0.5 ? '#dc2626' : missing > 0.2 ? '#d97706' : missing > 0 ? '#2563eb' : '#059669'
  const outlierColor = outlier > 0.1 ? '#dc2626' : outlier > 0.05 ? '#d97706' : '#059669'
  return (
    <div style={{ background: '#fff', border: '0.5px solid #e5e3db', borderRadius: 12, padding: '16px 18px', display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div style={{ fontWeight: 600, fontSize: 14, color: '#1a1a18', marginBottom: 3 }}>{profile.column}</div>
          <div style={{ fontSize: 11, color: '#9a9a94', fontFamily: 'monospace' }}>{profile.dtype} · {profile.cardinality_hint}</div>
        </div>
        <div style={{display:'flex',flexDirection:'column',alignItems:'flex-end',gap:4}}>
          <span style={{ fontSize: 11, fontWeight: 600, padding: '3px 9px', borderRadius: 99, background: meta.bg, color: meta.color, whiteSpace: 'nowrap' }}>{meta.label}</span>
          {(() => { const r = mlReadiness(profile, step); return <span style={{fontSize:10,fontWeight:600,padding:'2px 7px',borderRadius:99,background:r.bg,color:r.color}} title={r.tip}>{r.label}</span> })()}
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 11, color: '#6b7280', width: 60, flexShrink: 0 }}>Missing</span>
          <MiniBar value={missing} color={missingColor} />
          <span style={{ fontSize: 11, fontWeight: 600, color: missingColor, width: 36, textAlign: 'right' }}>{(missing * 100).toFixed(1)}%</span>
        </div>
        {isNumeric && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 11, color: '#6b7280', width: 60, flexShrink: 0 }}>Outliers</span>
            <MiniBar value={outlier} max={0.3} color={outlierColor} />
            <span style={{ fontSize: 11, fontWeight: 600, color: outlierColor, width: 36, textAlign: 'right' }}>{(outlier * 100).toFixed(1)}%</span>
          </div>
        )}
        {isNumeric && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 11, color: '#6b7280', width: 60, flexShrink: 0 }}>Skewness</span>
            <SkewnessGauge skewness={profile.skewness} />
          </div>
        )}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 11, color: '#6b7280', width: 60, flexShrink: 0 }}>Unique</span>
          <span style={{ fontSize: 12, color: '#374151' }}>{profile.unique_count ?? '—'} values</span>
        </div>
        {profile.sample_values?.length > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
            <span style={{ fontSize: 11, color: '#6b7280', flexShrink: 0 }}>Samples:</span>
            {profile.sample_values.slice(0, 3).map((v, i) => (
              <span key={i} style={{ fontSize: 11, padding: '1px 7px', background: '#f8fafc', border: '0.5px solid #e2e8f0', borderRadius: 4, fontFamily: 'monospace', color: '#374151' }}>
                {String(v).slice(0, 20)}
              </span>
            ))}
          </div>
        )}
      </div>
      {step?.rationale && (
        <div style={{ background: '#f8fafc', borderLeft: `3px solid ${meta.color}`, borderRadius: '0 6px 6px 0', padding: '8px 10px', fontSize: 12, color: '#5a5a56', lineHeight: 1.6 }}>
          {step.rationale}
        </div>
      )}
    </div>
  )
}

// ─── Action summary chart ─────────────────────────────────────────────────────
function ActionSummaryChart({ cleaningPlan }) {
  if (!cleaningPlan?.length) return null
  const counts = {}
  cleaningPlan.forEach(s => { const g = ACTION_META[s.action]?.label ?? s.action; counts[g] = (counts[g] ?? 0) + 1 })
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1])
  const maxVal = Math.max(...entries.map(e => e[1]))
  const BAR_H = 22, GAP = 6, LEFT = 110, W = 420
  const totalH = entries.length * (BAR_H + GAP)
  return (
    <div style={{ background: '#fff', border: '0.5px solid #e5e3db', borderRadius: 12, padding: '18px 20px', marginBottom: 20 }}>
      <p style={{ fontSize: 12, fontWeight: 600, color: '#6b7280', marginBottom: 14, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Actions applied</p>
      <svg width='100%' viewBox={`0 0 ${W} ${totalH}`} style={{ overflow: 'visible' }}>
        {entries.map(([label, count], i) => {
          const y = i * (BAR_H + GAP)
          const barW = (count / maxVal) * (W - LEFT - 50)
          const key = Object.keys(ACTION_META).find(k => ACTION_META[k].label === label) ?? 'keep'
          const color = ACTION_META[key]?.color ?? '#6b7280'
          const bg = ACTION_META[key]?.bg ?? '#f3f4f6'
          return (
            <g key={label}>
              <text x={LEFT - 8} y={y + BAR_H / 2 + 4} textAnchor='end' fontSize={11} fill='#6b7280'>{label}</text>
              <rect x={LEFT} y={y} width={W - LEFT - 50} height={BAR_H} rx={4} fill={bg} />
              <rect x={LEFT} y={y} width={barW} height={BAR_H} rx={4} fill={color} opacity={0.8} />
              <text x={LEFT + barW + 6} y={y + BAR_H / 2 + 4} fontSize={11} fontWeight={600} fill={color}>{count}</text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

// ─── Missing heatmap ──────────────────────────────────────────────────────────
function MissingHeatmap({ profiles }) {
  if (!profiles?.length) return null
  const withMissing = profiles.filter(p => (p.missing_pct ?? 0) > 0).sort((a, b) => b.missing_pct - a.missing_pct)
  if (!withMissing.length) return (
    <div style={{ background: '#f0fdf4', border: '0.5px solid #bbf7d0', borderRadius: 10, padding: '12px 16px', fontSize: 13, color: '#059669', marginBottom: 20 }}>
      ✓ No missing values detected across any column.
    </div>
  )
  return (
    <div style={{ background: '#fff', border: '0.5px solid #e5e3db', borderRadius: 12, padding: '18px 20px', marginBottom: 20 }}>
      <p style={{ fontSize: 12, fontWeight: 600, color: '#6b7280', marginBottom: 14, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Missing values by column</p>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {withMissing.map(p => {
          const pct = p.missing_pct ?? 0
          const color = pct > 0.5 ? '#dc2626' : pct > 0.2 ? '#d97706' : '#2563eb'
          return (
            <div key={p.column} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <span style={{ fontSize: 12, fontWeight: 500, width: 120, flexShrink: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.column}</span>
              <div style={{ flex: 1, background: '#f1f5f9', borderRadius: 4, height: 14, position: 'relative', overflow: 'hidden' }}>
                <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: `${pct * 100}%`, background: color, borderRadius: 4, opacity: 0.85 }} />
              </div>
              <span style={{ fontSize: 12, fontWeight: 600, color, width: 42, textAlign: 'right' }}>{(pct * 100).toFixed(1)}%</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ─── CSV Viewer ───────────────────────────────────────────────────────────────
const PAGE_SIZE = 50

function CSVViewer({ csvUrl }) {
  const [csvState, setCsvState] = useState({ status: 'idle', headers: [], rows: [], error: '' })
  const [selectedCols, setSelectedCols] = useState(new Set())
  const [page, setPage] = useState(0)
  const [colSearch, setColSearch] = useState('')
  const [rowSearch, setRowSearch] = useState('')
  const [showColPanel, setShowColPanel] = useState(false)

  useEffect(() => {
    if (!csvUrl) return
    setCsvState(s => ({ ...s, status: 'loading' }))
    fetch(csvUrl)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.text() })
      .then(text => {
        const { headers, rows } = parseCSV(text)
        setCsvState({ status: 'loaded', headers, rows, error: '' })
        setSelectedCols(new Set(headers.map((_, i) => i)))
      })
      .catch(e => setCsvState({ status: 'error', headers: [], rows: [], error: e.message }))
  }, [csvUrl])

  const { status, headers, rows, error } = csvState

  const filteredRows = rowSearch
    ? rows.filter(row => row.some(cell => String(cell ?? '').toLowerCase().includes(rowSearch.toLowerCase())))
    : rows

  const totalPages = Math.ceil(filteredRows.length / PAGE_SIZE)
  const pageRows = filteredRows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)
  const visibleColIdxs = headers.map((_, i) => i).filter(i => selectedCols.has(i))

  const toggleCol = i => setSelectedCols(prev => {
    const next = new Set(prev)
    next.has(i) ? next.delete(i) : next.add(i)
    return next
  })

  const toggleAll = () =>
    setSelectedCols(selectedCols.size === headers.length ? new Set() : new Set(headers.map((_, i) => i)))

  const panelCols = headers.map((h, i) => ({ h, i })).filter(({ h }) =>
    h.toLowerCase().includes(colSearch.toLowerCase())
  )

  if (!csvUrl) return <p style={{ fontSize: 13, color: '#9a9a94' }}>No CSV URL available.</p>
  if (status === 'loading') return (
    <div style={{ padding: '3rem', textAlign: 'center', color: '#9a9a94', fontSize: 14 }}>⏳ Loading cleaned data…</div>
  )
  if (status === 'error') return (
    <div style={{ padding: '1rem', background: '#fef2f2', border: '0.5px solid #fecaca', borderRadius: 8, color: '#dc2626', fontSize: 13 }}>
      Failed to load CSV: {error}
    </div>
  )
  if (status === 'idle') return (
    <div style={{ padding: '2rem', textAlign: 'center' }}>
      <button onClick={() => setCsvState(s => ({ ...s, status: 'loading' }))}
        style={{ padding: '10px 20px', background: '#1a1a18', color: '#fff', border: 'none', borderRadius: 8, cursor: 'pointer', fontSize: 13 }}>
        Load data
      </button>
    </div>
  )

  return (
    <div>
      {/* Toolbar */}
      <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 14, flexWrap: 'wrap' }}>
        <input
          value={rowSearch}
          onChange={e => { setRowSearch(e.target.value); setPage(0) }}
          placeholder='Search rows…'
          style={{ padding: '7px 12px', borderRadius: 7, border: '0.5px solid #d3d1c7', fontSize: 13, flex: '1 1 180px', minWidth: 0, outline: 'none' }}
        />
        <button onClick={() => setShowColPanel(v => !v)} style={{
          padding: '7px 14px', borderRadius: 7, fontSize: 13, cursor: 'pointer',
          border: '0.5px solid #d3d1c7',
          background: showColPanel ? '#1a1a18' : '#fff',
          color: showColPanel ? '#fff' : '#1a1a18',
          fontWeight: 500, whiteSpace: 'nowrap',
        }}>
          ⚙ Columns ({selectedCols.size}/{headers.length})
        </button>
        <button
          onClick={() => exportCSV(headers, rows, selectedCols)}
          disabled={selectedCols.size === 0}
          style={{
            padding: '7px 14px', borderRadius: 7, fontSize: 13,
            cursor: selectedCols.size ? 'pointer' : 'not-allowed',
            border: 'none',
            background: selectedCols.size ? '#059669' : '#d1d5db',
            color: '#fff', fontWeight: 600, whiteSpace: 'nowrap',
          }}>
          ↓ Download selected columns
        </button>
        <span style={{ fontSize: 12, color: '#9a9a94', whiteSpace: 'nowrap' }}>
          {filteredRows.length.toLocaleString()} rows · {selectedCols.size} cols visible
        </span>
      </div>

      {/* Column panel */}
      {showColPanel && (
        <div style={{ background: '#fff', border: '0.5px solid #e5e3db', borderRadius: 10, padding: '14px 16px', marginBottom: 14 }}>
          <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 10 }}>
            <input
              value={colSearch}
              onChange={e => setColSearch(e.target.value)}
              placeholder='Filter columns…'
              style={{ padding: '6px 10px', borderRadius: 6, border: '0.5px solid #d3d1c7', fontSize: 12, flex: 1, outline: 'none' }}
            />
            <button onClick={toggleAll} style={{ padding: '6px 12px', borderRadius: 6, fontSize: 12, border: '0.5px solid #d3d1c7', cursor: 'pointer', background: '#f9f9f8', whiteSpace: 'nowrap' }}>
              {selectedCols.size === headers.length ? 'Deselect all' : 'Select all'}
            </button>
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 7 }}>
            {panelCols.map(({ h, i }) => (
              <label key={i} style={{
                display: 'flex', alignItems: 'center', gap: 5, cursor: 'pointer',
                padding: '4px 10px', borderRadius: 99, fontSize: 12,
                background: selectedCols.has(i) ? '#dbeafe' : '#f3f4f6',
                color: selectedCols.has(i) ? '#1d4ed8' : '#6b7280',
                border: `0.5px solid ${selectedCols.has(i) ? '#93c5fd' : '#e5e7eb'}`,
                userSelect: 'none',
              }}>
                <input type='checkbox' checked={selectedCols.has(i)} onChange={() => toggleCol(i)}
                  style={{ width: 12, height: 12, accentColor: '#2563eb' }} />
                {h || `col_${i}`}
              </label>
            ))}
          </div>
        </div>
      )}

      {/* Table */}
      <div style={{ overflowX: 'auto', border: '0.5px solid #e5e3db', borderRadius: 10, background: '#fff' }}>
        <table style={{ borderCollapse: 'collapse', fontSize: 12, minWidth: '100%' }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 10px', textAlign: 'center', color: '#9a9a94', borderBottom: '0.5px solid #e5e3db', fontWeight: 500, minWidth: 44 }}>#</th>
              {visibleColIdxs.map(i => (
                <th key={i} style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '0.5px solid #e5e3db', fontWeight: 600, color: '#374151', whiteSpace: 'nowrap' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                    {headers[i] || `col_${i}`}
                    <button onClick={() => toggleCol(i)} title='Hide column'
                      style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#cbd5e1', fontSize: 10, padding: 0, lineHeight: 1 }}>
                      ✕
                    </button>
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {pageRows.map((row, ri) => (
              <tr key={ri} style={{ borderBottom: '0.5px solid #f1f5f9', background: ri % 2 === 0 ? '#fff' : '#fafafa' }}>
                <td style={{ padding: '6px 10px', textAlign: 'center', color: '#9a9a94', fontSize: 11 }}>
                  {page * PAGE_SIZE + ri + 1}
                </td>
                {visibleColIdxs.map(i => {
                  const val = row[i] ?? ''
                  const isEmpty = val === '' || val === 'nan' || val === 'NaN' || val === 'null'
                  return (
                    <td key={i} style={{
                      padding: '6px 12px',
                      color: isEmpty ? '#d1d5db' : '#1a1a18',
                      fontFamily: 'monospace',
                      maxWidth: 180,
                      overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                    }}>
                      {isEmpty ? '—' : val}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6, marginTop: 14, flexWrap: 'wrap' }}>
          <PagBtn label='«' disabled={page === 0} onClick={() => setPage(0)} />
          <PagBtn label='‹' disabled={page === 0} onClick={() => setPage(p => p - 1)} />
          {(() => {
            const start = Math.max(0, Math.min(page - 3, totalPages - 7))
            return Array.from({ length: Math.min(7, totalPages) }, (_, k) => start + k).map(p => (
              <PagBtn key={p} label={p + 1} active={p === page} onClick={() => setPage(p)} />
            ))
          })()}
          <PagBtn label='›' disabled={page === totalPages - 1} onClick={() => setPage(p => p + 1)} />
          <PagBtn label='»' disabled={page === totalPages - 1} onClick={() => setPage(totalPages - 1)} />
          <span style={{ fontSize: 12, color: '#9a9a94', marginLeft: 4 }}>
            Page {page + 1}/{totalPages} · rows {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, filteredRows.length)}
          </span>
        </div>
      )}
    </div>
  )
}

function PagBtn({ label, onClick, disabled, active }) {
  return (
    <button onClick={onClick} disabled={disabled} style={{
      padding: '5px 10px', borderRadius: 6, fontSize: 13, cursor: disabled ? 'not-allowed' : 'pointer',
      border: '0.5px solid', borderColor: active ? '#1a1a18' : '#d3d1c7',
      background: active ? '#1a1a18' : '#fff',
      color: active ? '#fff' : disabled ? '#d1d5db' : '#1a1a18',
    }}>
      {label}
    </button>
  )
}

// ─── Main ResultsPage ─────────────────────────────────────────────────────────
export default function ResultsPage({ results, onReset, apiKey = '' }) {
  const [tab, setTab] = useState('overview')
  const [search, setSearch] = useState('')

  const score      = results?.final_score?.overall ?? 0
  const missingPct = 1 - (results?.final_score?.missing_score ?? 1)
  const scoreColor = score >= 0.9 ? '#0F6E56' : score >= 0.7 ? '#854F0B' : '#A32D2D'
  const scoreBg    = score >= 0.9 ? '#E1F5EE' : score >= 0.7 ? '#FAEEDA' : '#FCEBEB'

  const rowsBefore = results?.shape_before?.[0] ?? '—'
  const colsBefore = results?.shape_before?.[1] ?? '—'
  const rowsAfter  = results?.shape_after?.[0] ?? '—'
  const colsAfter  = results?.shape_after?.[1] ?? '—'

  const stepByCol = {}
  ;(results?.cleaning_plan ?? []).forEach(s => { stepByCol[s.column] = s })

  const profiles = results?.column_profiles ?? []
  const filtered = profiles.filter(p => p.column.toLowerCase().includes(search.toLowerCase()))

  const TABS = [
    ['overview',  'Overview'],
    ['viewer',    '📊 Data Viewer'],
    ['histograms','📉 Distributions'],
    ['alerts',    '⚠ Alerts'],
    ['insights',  '🔍 Column Insights'],
    ['reasoning', 'Agent Reasoning'],
    ['correlation','🔗 Correlation'],
    ['ml',         '🎯 ML Readiness'],
    ['lineage',    '🔀 Data Lineage'],
    ['history',   'Iteration History'],
  ]

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: '3rem 1.5rem', fontFamily: 'system-ui, -apple-system, sans-serif' }}>

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '2rem' }}>
        <div>
          <p style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.12em', textTransform: 'uppercase', color: '#9a9a94', marginBottom: 4 }}>Pipeline complete</p>
          <h1 style={{ fontSize: 22, fontWeight: 600, color: '#1a1a18' }}>{results?.filename ?? 'Results'}</h1>
        </div>
        <div style={{display:'flex',gap:8}}>
          <button onClick={() => {
            const url = `${window.location.origin}/report/${results?.run_id}`
            navigator.clipboard.writeText(url).then(() => alert('Report link copied!'))
          }} style={{padding:'8px 14px',borderRadius:8,fontSize:13,border:'0.5px solid #d3d1c7',background:'#fff',cursor:'pointer',color:'#1a1a18'}}>
            Share ↗
          </button>
          <button onClick={onReset} style={{ padding: '8px 16px', borderRadius: 8, fontSize: 13, border: '0.5px solid #d3d1c7', background: '#fff', cursor: 'pointer', color: '#5a5a56' }}>
            ← Clean another file
          </button>
        </div>
      </div>

      {/* Metric cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 10, marginBottom: '2rem' }}>
        {[
          { label: 'Quality score',     value: `${Math.round(score * 100)}%`,         color: scoreColor, bg: scoreBg },
          { label: 'Iterations',        value: results?.iterations_completed ?? '—',   color: '#185FA5',  bg: '#E6F1FB' },
          { label: 'Columns cleaned',   value: results?.cleaning_plan?.length ?? '—', color: '#534AB7',  bg: '#EEEDFE' },
          { label: 'Missing remaining', value: `${Math.round(missingPct * 100)}%`,     color: '#854F0B',  bg: '#FAEEDA' },
        ].map(card => (
          <div key={card.label} style={{ padding: '14px 16px', background: '#fff', border: '0.5px solid #e5e3db', borderRadius: 10 }}>
            <p style={{ fontSize: 11, color: '#9a9a94', marginBottom: 6 }}>{card.label}</p>
            <p style={{ fontSize: 22, fontWeight: 600, color: card.color, background: card.bg, display: 'inline-block', padding: '2px 10px', borderRadius: 99 }}>{card.value}</p>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '0.5px solid #e5e3db', overflowX: 'auto' }}>
        {TABS.map(([key, label]) => (
          <button key={key} onClick={() => setTab(key)} style={{
            padding: '8px 14px', fontSize: 13, fontWeight: tab === key ? 500 : 400,
            border: 'none', background: 'none', cursor: 'pointer', whiteSpace: 'nowrap',
            color: tab === key ? '#1a1a18' : '#9a9a94',
            borderBottom: tab === key ? '2px solid #1a1a18' : '2px solid transparent',
            marginBottom: -1,
          }}>{label}</button>
        ))}
      </div>

      {/* ── Overview ── */}
      {tab === 'overview' && (
        <div>
          <AISummaryCard runId={results?.run_id} apiKey={apiKey} results={results} />
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginBottom: 20 }}>
            {[
              { label: 'Shape before',   value: `${rowsBefore} rows × ${colsBefore} cols` },
              { label: 'Shape after',    value: `${rowsAfter} rows × ${colsAfter} cols` },
              { label: 'Outlier score',  value: `${Math.round((results?.final_score?.outlier_score ?? 0) * 100)}%` },
              { label: 'Skewness score', value: `${Math.round((results?.final_score?.skewness_score ?? 0) * 100)}%` },
            ].map(item => (
              <div key={item.label} style={{ padding: '12px 16px', background: '#fff', border: '0.5px solid #e5e3db', borderRadius: 10 }}>
                <p style={{ fontSize: 11, color: '#9a9a94', marginBottom: 4 }}>{item.label}</p>
                <p style={{ fontSize: 15, fontWeight: 600, color: '#1a1a18' }}>{item.value}</p>
              </div>
            ))}
          </div>
          {profiles.length > 0 && (
            <div style={{ overflowX: 'auto', marginBottom: 20 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f5f4f0' }}>
                    {['Column', 'Type', 'Missing', 'Outlier ratio', 'Skewness', 'Cardinality'].map(h => (
                      <th key={h} style={{ padding: '8px 12px', textAlign: 'left', fontWeight: 600, color: '#5a5a56', borderBottom: '0.5px solid #e5e3db' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {profiles.map((p, i) => (
                    <tr key={i} style={{ borderBottom: '0.5px solid #e5e3db' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{p.column}</td>
                      <td style={{ padding: '8px 12px', color: '#9a9a94', fontFamily: 'monospace', fontSize: 12 }}>{p.dtype}</td>
                      <td style={{ padding: '8px 12px' }}>{p.missing_pct != null ? `${(p.missing_pct * 100).toFixed(1)}%` : '—'}</td>
                      <td style={{ padding: '8px 12px' }}>{p.outlier_ratio != null ? `${(p.outlier_ratio * 100).toFixed(1)}%` : '—'}</td>
                      <td style={{ padding: '8px 12px' }}>{p.skewness != null ? p.skewness.toFixed(2) : '—'}</td>
                      <td style={{ padding: '8px 12px' }}><span style={{ padding: '2px 8px', borderRadius: 99, background: '#f5f4f0', fontSize: 11 }}>{p.cardinality_hint}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            {results?.csv_download_url && (
              <a href={results.csv_download_url} download style={{ padding: '10px 20px', background: '#1a1a18', color: '#fff', borderRadius: 8, fontSize: 13, fontWeight: 500, textDecoration: 'none' }}>
                ↓ Download cleaned CSV
              </a>
            )}
            {results?.report_download_url && (
              <a href={results.report_download_url} download style={{ padding: '10px 20px', background: '#fff', color: '#1a1a18', borderRadius: 8, fontSize: 13, fontWeight: 500, textDecoration: 'none', border: '0.5px solid #d3d1c7' }}>
                ↓ Download audit report
              </a>
            )}
            {results?.run_id && (
              <a href={`/api/download/${results.run_id}/parquet`} download style={{ padding: '10px 20px', background: '#fff', color: '#1a1a18', borderRadius: 8, fontSize: 13, fontWeight: 500, textDecoration: 'none', border: '0.5px solid #d3d1c7' }}>
                ↓ Download Parquet
              </a>
            )}
          </div>
        </div>
      )}

      {/* ── Distributions ── */}
      {tab === 'histograms' && (() => {
        const beforeByCol = {}
        const afterByCol  = {}
        ;(results?.column_profiles || []).forEach(p => { beforeByCol[p.column] = p })
        ;(results?.cleaned_column_profiles || []).forEach(p => { afterByCol[p.column] = p })

        const numericCols = (results?.column_profiles || []).filter(p => p.histogram)
        const catCols     = (results?.column_profiles || []).filter(p => p.value_counts?.length)

        return (
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 20, flexWrap: 'wrap' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <div style={{ width: 12, height: 10, background: '#94a3b8', borderRadius: 1, opacity: 0.6 }} />
                <span style={{ fontSize: 12, color: '#6b7280' }}>Before cleaning</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <div style={{ width: 12, height: 10, background: '#2563eb', borderRadius: 1, opacity: 0.75 }} />
                <span style={{ fontSize: 12, color: '#6b7280' }}>After cleaning</span>
              </div>
            </div>

            {numericCols.length > 0 && (
              <div style={{ marginBottom: 28 }}>
                <p style={{ fontSize: 12, fontWeight: 600, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 14 }}>Numeric columns</p>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 12 }}>
                  {numericCols.map((p, i) => {
                    const after = afterByCol[p.column]
                    const step  = (results?.cleaning_plan || []).find(s => s.column === p.column)
                    const meta  = ACTION_META[step?.action] ?? ACTION_META.keep
                    const ns    = p.numeric_summary || {}
                    const nsA   = after?.numeric_summary || {}
                    return (
                      <div key={i} style={{ background: '#fff', border: '0.5px solid #e5e3db', borderRadius: 10, padding: '14px 16px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                          <span style={{ fontSize: 13, fontWeight: 600, color: '#1a1a18' }}>{p.column}</span>
                          <span style={{ fontSize: 10, fontWeight: 600, padding: '2px 8px', borderRadius: 99, background: meta.bg, color: meta.color }}>{meta.label}</span>
                        </div>
                        <HistogramChart before={p.histogram} after={after?.histogram} column={p.column} action={step?.action} />
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginTop: 10 }}>
                          {[
                            { label: 'Mean',   before: ns.mean,   after: nsA.mean   },
                            { label: 'Median', before: ns.median, after: nsA.median },
                            { label: 'Std',    before: ns.std,    after: nsA.std    },
                            { label: 'Skew',   before: p.skewness, after: after?.skewness },
                          ].map(m => (
                            <div key={m.label} style={{ background: '#f8fafc', borderRadius: 6, padding: '6px 8px' }}>
                              <p style={{ fontSize: 10, color: '#9a9a94', marginBottom: 2 }}>{m.label}</p>
                              <div style={{ display: 'flex', gap: 4, alignItems: 'baseline' }}>
                                <span style={{ fontSize: 11, color: '#94a3b8' }}>{m.before != null ? Number(m.before).toFixed(2) : '—'}</span>
                                {m.after != null && m.before != null && (
                                  <>
                                    <span style={{ fontSize: 9, color: '#cbd5e1' }}>→</span>
                                    <span style={{ fontSize: 11, fontWeight: 600, color: '#2563eb' }}>{Number(m.after).toFixed(2)}</span>
                                  </>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {catCols.length > 0 && (
              <div>
                <p style={{ fontSize: 12, fontWeight: 600, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 14 }}>Categorical columns — top values</p>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 12 }}>
                  {catCols.map((p, i) => {
                    const step = (results?.cleaning_plan || []).find(s => s.column === p.column)
                    const meta = ACTION_META[step?.action] ?? ACTION_META.keep
                    return (
                      <div key={i} style={{ background: '#fff', border: '0.5px solid #e5e3db', borderRadius: 10, padding: '14px 16px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                          <span style={{ fontSize: 13, fontWeight: 600, color: '#1a1a18' }}>{p.column}</span>
                          <span style={{ fontSize: 10, fontWeight: 600, padding: '2px 8px', borderRadius: 99, background: meta.bg, color: meta.color }}>{meta.label}</span>
                        </div>
                        <ValueCountsBar valueCounts={p.value_counts} />
                        <p style={{ fontSize: 11, color: '#9a9a94', marginTop: 8 }}>{p.unique_count} unique values · {p.cardinality_hint} cardinality</p>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}
          </div>
        )
      })()}

      {/* ── Alerts ── */}
      {tab === 'alerts' && (
        <div>
          <p style={{ fontSize: 13, color: '#6b7280', marginBottom: 16 }}>
            Data quality issues detected on the <strong>original</strong> dataset, and what the agent did to fix each one.
          </p>
          <AlertsPanel profiles={results?.column_profiles} cleaningPlan={results?.cleaning_plan} />
        </div>
      )}

      {/* ── Data Viewer ── */}
      {tab === 'viewer' && (
        <div>
          <p style={{ fontSize: 13, color: '#6b7280', marginBottom: 16 }}>
            Browse the full cleaned dataset below. Use <strong>⚙ Columns</strong> to show/hide columns or click <strong>✕</strong> on any column header to hide it instantly. Then download only the columns you want.
          {results?.duplicate_count > 0 && (
            <div style={{marginTop:10,padding:'8px 14px',background:'#fffbeb',border:'0.5px solid #fde68a',borderRadius:8,display:'flex',alignItems:'center',justifyContent:'space-between',gap:10,flexWrap:'wrap'}}>
              <span style={{fontSize:13,color:'#92400e'}}>⚠ <strong>{results.duplicate_count}</strong> duplicate rows detected in cleaned dataset</span>
              <button onClick={() => {
                const link = document.createElement('a')
                link.href = results.csv_download_url.replace('/csv', '/csv-dedup')
                link.click()
              }} style={{fontSize:12,padding:'4px 12px',borderRadius:6,border:'0.5px solid #d97706',background:'#fff',color:'#92400e',cursor:'pointer',fontWeight:600}}>
                Download deduplicated ↓
              </button>
            </div>
          )}
          </p>
          <CSVViewer csvUrl={results?.csv_download_url} />
        </div>
      )}

      {/* ── Column Insights ── */}
      {tab === 'insights' && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
            <ActionSummaryChart cleaningPlan={results?.cleaning_plan} />
            <MissingHeatmap profiles={profiles} />
          </div>
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder={`Search ${profiles.length} columns…`}
            style={{ width: '100%', boxSizing: 'border-box', padding: '9px 14px', borderRadius: 8, border: '0.5px solid #d3d1c7', fontSize: 13, background: '#fff', color: '#1a1a18', marginBottom: 16, outline: 'none' }}
          />
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 12 }}>
            {filtered.map((p, i) => <ColumnCard key={i} profile={p} step={stepByCol[p.column]} />)}
          </div>
          {filtered.length === 0 && (
            <p style={{ fontSize: 13, color: '#9a9a94', textAlign: 'center', padding: '2rem' }}>No columns match "{search}"</p>
          )}
        </div>
      )}

      {/* ── Agent Reasoning ── */}
      {tab === 'reasoning' && (() => {
        const plan = results?.cleaning_plan ?? []
        if (!plan.length) return <p style={{ fontSize: 13, color: '#9a9a94' }}>No cleaning plan available.</p>

        // Define category groups with display order
        const CATEGORIES = [
          {
            key: 'impute',
            label: 'Missing Value Imputation',
            icon: '💧',
            color: '#2563eb',
            bg: '#eff6ff',
            border: '#bfdbfe',
            actions: ['impute_mean','impute_median','impute_mode','impute_constant'],
            desc: 'Columns with missing values — filled using statistical methods',
          },
          {
            key: 'outlier',
            label: 'Outlier Handling',
            icon: '📐',
            color: '#d97706',
            bg: '#fffbeb',
            border: '#fde68a',
            actions: ['clip_iqr','winsorise'],
            desc: 'Columns with extreme values clipped to statistical bounds',
          },
          {
            key: 'encode',
            label: 'Categorical Encoding',
            icon: '🔤',
            color: '#7c3aed',
            bg: '#f5f3ff',
            border: '#ddd6fe',
            actions: ['encode_onehot','encode_ordinal','encode_binary'],
            desc: 'Text/categorical columns converted to numeric representation',
          },
          {
            key: 'scale',
            label: 'Scaling & Normalisation',
            icon: '⚖️',
            color: '#059669',
            bg: '#f0fdf4',
            border: '#a7f3d0',
            actions: ['scale_standard','scale_minmax','scale_robust'],
            desc: 'Numeric columns rescaled for consistent model performance',
          },
          {
            key: 'transform',
            label: 'Distribution Transforms',
            icon: '📈',
            color: '#0891b2',
            bg: '#ecfeff',
            border: '#a5f3fc',
            actions: ['log_transform','sqrt_transform'],
            desc: 'Skewed distributions corrected via mathematical transforms',
          },
          {
            key: 'drop',
            label: 'Dropped Columns',
            icon: '🗑️',
            color: '#dc2626',
            bg: '#fff1f2',
            border: '#fecdd3',
            actions: ['drop_column'],
            desc: 'Columns removed due to high null rate or no useful signal',
          },
          {
            key: 'keep',
            label: 'No Action Needed',
            icon: '✓',
            color: '#6b7280',
            bg: '#f9fafb',
            border: '#e5e7eb',
            actions: ['keep'],
            desc: 'Columns already clean — agent left them unchanged',
          },
        ]

        // Group steps by category
        const grouped = {}
        plan.forEach(step => {
          const cat = CATEGORIES.find(c => c.actions.includes(step.action))
          const key = cat?.key ?? 'keep'
          if (!grouped[key]) grouped[key] = []
          grouped[key].push(step)
        })

        return (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {CATEGORIES.filter(cat => grouped[cat.key]?.length > 0).map(cat => {
              const steps = grouped[cat.key]
              return (
                <div key={cat.key} style={{ border: `1px solid ${cat.border}`, borderRadius: 12, overflow: 'hidden' }}>
                  {/* Category header */}
                  <div style={{ background: cat.bg, padding: '12px 18px', display: 'flex', alignItems: 'center', gap: 10, borderBottom: `1px solid ${cat.border}` }}>
                    <span style={{ fontSize: 18 }}>{cat.icon}</span>
                    <div style={{ flex: 1 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{ fontSize: 14, fontWeight: 700, color: cat.color }}>{cat.label}</span>
                        <span style={{ fontSize: 11, fontWeight: 600, padding: '1px 8px', borderRadius: 99, background: cat.color, color: '#fff' }}>
                          {steps.length} {steps.length === 1 ? 'column' : 'columns'}
                        </span>
                      </div>
                      <p style={{ fontSize: 11, color: '#6b7280', margin: '2px 0 0' }}>{cat.desc}</p>
                    </div>
                    {/* Column name pills preview */}
                    <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap', justifyContent: 'flex-end', maxWidth: 280 }}>
                      {steps.map((s, i) => (
                        <span key={i} style={{ fontSize: 11, padding: '2px 8px', borderRadius: 99, background: '#fff', border: `1px solid ${cat.border}`, color: cat.color, fontWeight: 500 }}>
                          {s.column}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Step rows */}
                  <div style={{ background: '#fff' }}>
                    {steps.map((step, si) => {
                      const meta = ACTION_META[step.action] ?? ACTION_META.keep
                      return (
                        <div key={si} style={{
                          display: 'grid',
                          gridTemplateColumns: '160px 1fr',
                          borderBottom: si < steps.length - 1 ? '0.5px solid #f1f5f9' : 'none',
                        }}>
                          {/* Left: column name + action badge */}
                          <div style={{
                            padding: '12px 14px',
                            borderRight: '0.5px solid #f1f5f9',
                            background: '#fafafa',
                            display: 'flex', flexDirection: 'column', gap: 6, justifyContent: 'center',
                          }}>
                            <span style={{ fontSize: 13, fontWeight: 600, color: '#1a1a18' }}>{step.column}</span>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 5, flexWrap: 'wrap' }}>
                              <span style={{ fontSize: 10, fontWeight: 600, padding: '2px 7px', borderRadius: 99, background: meta.bg, color: meta.color }}>
                                {meta.label}
                              </span>
                              {step.params && Object.keys(step.params).length > 0 && (
                                <span style={{ fontSize: 10, padding: '2px 6px', borderRadius: 4, background: '#f3f4f6', color: '#6b7280', fontFamily: 'monospace' }}>
                                  {JSON.stringify(step.params)}
                                </span>
                              )}
                            </div>
                          </div>
                          {/* Right: rationale */}
                          <div style={{ padding: '12px 16px', display: 'flex', alignItems: 'center' }}>
                            <p style={{ fontSize: 13, color: '#4b5563', lineHeight: 1.6, margin: 0 }}>{step.rationale}</p>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )
            })}
          </div>
        )
      })()}

      {/* ── Correlation ── */}
      {tab === 'correlation' && (
        <div>
          <p style={{fontSize:13,color:'#6b7280',marginBottom:16}}>Pearson correlation between numeric columns. Blue = positive, red = negative. Intensity = strength.</p>
          <CorrelationHeatmap correlation={results?.correlation} />
        </div>
      )}

      {/* ── ML Readiness ── */}
      {tab === 'ml' && (() => {
        const plan = results?.cleaning_plan ?? []
        const stepByCol = {}
        plan.forEach(s => { stepByCol[s.column] = s })
        const profiles = results?.column_profiles ?? []
        const ready   = profiles.filter(p => mlReadiness(p, stepByCol[p.column]).level === 'ready')
        const warn    = profiles.filter(p => mlReadiness(p, stepByCol[p.column]).level === 'warn')
        const risk    = profiles.filter(p => mlReadiness(p, stepByCol[p.column]).level === 'risk')
        const dropped = profiles.filter(p => mlReadiness(p, stepByCol[p.column]).level === 'dropped')
        const GROUPS = [
          { label: '✓ ML Ready', cols: ready,   color:'#059669', bg:'#d1fae5', border:'#a7f3d0' },
          { label: '⚠ Review',   cols: warn,    color:'#d97706', bg:'#fef3c7', border:'#fde68a' },
          { label: '✗ High Risk',cols: risk,    color:'#dc2626', bg:'#fee2e2', border:'#fecaca' },
          { label: '— Dropped',  cols: dropped, color:'#6b7280', bg:'#f3f4f6', border:'#e5e7eb' },
        ].filter(g => g.cols.length > 0)

        return (
          <div>
            <div style={{display:'flex',gap:10,marginBottom:20,flexWrap:'wrap'}}>
              {GROUPS.map(g => (
                <div key={g.label} style={{padding:'8px 14px',background:g.bg,border:`0.5px solid ${g.border}`,borderRadius:8,fontSize:12,fontWeight:600,color:g.color}}>
                  {g.label} ({g.cols.length})
                </div>
              ))}
            </div>
            {GROUPS.map(g => (
              <div key={g.label} style={{marginBottom:20}}>
                <p style={{fontSize:12,fontWeight:600,color:'#6b7280',textTransform:'uppercase',letterSpacing:'0.08em',marginBottom:10}}>{g.label}</p>
                <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(260px,1fr))',gap:8}}>
                  {g.cols.map((p, i) => {
                    const step = stepByCol[p.column]
                    const r = mlReadiness(p, step)
                    const meta = ACTION_META[step?.action] ?? ACTION_META.keep
                    const issues = r.tip !== 'No issues detected' && r.tip !== 'Column was removed' ? r.tip.split(', ') : []
                    return (
                      <div key={i} style={{background:'#fff',border:`0.5px solid ${g.border}`,borderRadius:10,padding:'12px 14px'}}>
                        <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:6}}>
                          <span style={{fontSize:13,fontWeight:600,color:'#1a1a18'}}>{p.column}</span>
                          <span style={{fontSize:10,padding:'2px 8px',borderRadius:99,background:meta.bg,color:meta.color,fontWeight:600}}>{meta.label}</span>
                        </div>
                        <div style={{fontSize:11,color:'#6b7280',marginBottom:6}}>{p.dtype} · {p.cardinality_hint}</div>
                        {issues.length > 0 && (
                          <div style={{display:'flex',flexWrap:'wrap',gap:4}}>
                            {issues.map((iss,j) => (
                              <span key={j} style={{fontSize:10,padding:'1px 7px',borderRadius:99,background:g.bg,color:g.color,border:`0.5px solid ${g.border}`}}>{iss}</span>
                            ))}
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              </div>
            ))}
          </div>
        )
      })()}


      {/* ── Data Lineage ── */}
      {tab === 'lineage' && (() => {
        const plan = results?.cleaning_plan ?? []
        const profiles = results?.column_profiles ?? []
        if (!plan.length) return <p style={{fontSize:13,color:'#9a9a94'}}>No cleaning plan available.</p>

        const ACTION_GROUPS = {
          impute:    { color:'#2563eb', label:'Imputed' },
          encode:    { color:'#7c3aed', label:'Encoded' },
          scale:     { color:'#059669', label:'Scaled' },
          transform: { color:'#0891b2', label:'Transformed' },
          clip:      { color:'#d97706', label:'Clipped' },
          drop:      { color:'#dc2626', label:'Dropped' },
          keep:      { color:'#9ca3af', label:'Unchanged' },
        }
        const getGroup = action => {
          if (action.startsWith('impute')) return 'impute'
          if (action.startsWith('encode')) return 'encode'
          if (action.startsWith('scale')) return 'scale'
          if (action.includes('transform')) return 'transform'
          if (action === 'clip_iqr' || action === 'winsorise') return 'clip'
          if (action === 'drop_column') return 'drop'
          return 'keep'
        }

        const profileMap = {}
        profiles.forEach(p => { profileMap[p.column] = p })

        return (
          <div>
            <p style={{fontSize:13,color:'#6b7280',marginBottom:20}}>
              Each column flows from its raw state (left) through the agent's transformation to the cleaned output (right).
            </p>
            <div style={{display:'flex',flexDirection:'column',gap:6}}>
              {plan.map((step, i) => {
                const grp = ACTION_GROUPS[getGroup(step.action)]
                const p   = profileMap[step.column] || {}
                const isDrop = step.action === 'drop_column'
                const meta = ACTION_META[step.action] ?? ACTION_META.keep
                return (
                  <div key={i} style={{display:'grid',gridTemplateColumns:'1fr 40px 1fr',alignItems:'center',gap:0,opacity: isDrop ? 0.5 : 1}}>
                    {/* Source node */}
                    <div style={{background:'#fff',border:'0.5px solid #e5e3db',borderRadius:'8px 0 0 8px',padding:'8px 12px',borderRight:'none'}}>
                      <div style={{fontSize:13,fontWeight:600,color:'#1a1a18'}}>{step.column}</div>
                      <div style={{fontSize:11,color:'#9a9a94',marginTop:2}}>
                        {p.dtype || '?'} · {p.missing_pct > 0 ? `${(p.missing_pct*100).toFixed(0)}% missing` : 'complete'}
                        {p.skewness != null && Math.abs(p.skewness) > 1 ? ` · skew ${p.skewness?.toFixed(1)}` : ''}
                      </div>
                    </div>
                    {/* Arrow + action */}
                    <div style={{display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center',background:grp.color,height:'100%',minHeight:46}}>
                      <div style={{fontSize:8,color:'rgba(255,255,255,0.9)',fontWeight:600,textAlign:'center',padding:'0 2px',lineHeight:1.3}}>→</div>
                    </div>
                    {/* Output node */}
                    <div style={{background: isDrop ? '#fef2f2' : '#f0fdf4', border:`0.5px solid ${isDrop ? '#fecaca' : '#bbf7d0'}`,borderRadius:'0 8px 8px 0',padding:'8px 12px',borderLeft:'none'}}>
                      {isDrop ? (
                        <div style={{fontSize:12,color:'#dc2626',fontWeight:600}}>✕ dropped</div>
                      ) : (
                        <>
                          <div style={{fontSize:12,fontWeight:600,color: grp.color}}>{meta.label}</div>
                          <div style={{fontSize:11,color:'#6b7280',marginTop:2,lineHeight:1.4}}>{(step.rationale || '').slice(0,70)}{step.rationale?.length > 70 ? '…' : ''}</div>
                        </>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
            {/* Legend */}
            <div style={{marginTop:20,display:'flex',gap:10,flexWrap:'wrap'}}>
              {Object.entries(ACTION_GROUPS).map(([k,v]) => (
                <div key={k} style={{display:'flex',alignItems:'center',gap:5,fontSize:11,color:'#6b7280'}}>
                  <div style={{width:12,height:12,borderRadius:2,background:v.color}}/>
                  {v.label}
                </div>
              ))}
            </div>
          </div>
        )
      })()}

      {/* ── Iteration History ── */}
      {tab === 'history' && (
        <div>
          {results?.quality_history?.length > 0 ? results.quality_history.map((q, i) => (
            <div key={i} style={{ padding: '14px 16px', background: '#fff', border: '0.5px solid #e5e3db', borderRadius: 10, marginBottom: 8 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 10 }}>
                <span style={{ fontSize: 12, fontWeight: 600, color: '#9a9a94', minWidth: 70 }}>Iteration {i + 1}</span>
                <div style={{ flex: 1, background: '#f5f4f0', borderRadius: 99, height: 6 }}>
                  <div style={{ width: `${q.overall * 100}%`, background: '#1D9E75', height: 6, borderRadius: 99 }} />
                </div>
                <span style={{ fontSize: 13, fontWeight: 500, color: '#1a1a18', minWidth: 40, textAlign: 'right' }}>{Math.round(q.overall * 100)}%</span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8 }}>
                {[{ label: 'Missing', value: q.missing_score }, { label: 'Outlier', value: q.outlier_score }, { label: 'Skewness', value: q.skewness_score }].map(m => (
                  <div key={m.label} style={{ background: '#f8fafc', borderRadius: 8, padding: '8px 10px' }}>
                    <p style={{ fontSize: 11, color: '#9a9a94', marginBottom: 2 }}>{m.label}</p>
                    <p style={{ fontSize: 13, fontWeight: 600, color: '#1a1a18' }}>{Math.round((m.value ?? 0) * 100)}%</p>
                  </div>
                ))}
              </div>
            </div>
          )) : <p style={{ fontSize: 13, color: '#9a9a94' }}>No iteration history available.</p>}
        </div>
      )}

    </div>
  )
}