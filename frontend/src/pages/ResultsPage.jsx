import React, { useState } from 'react'

export default function ResultsPage({ results, onReset }) {
  const [tab, setTab] = useState('overview')

  // results shape (from backend — implement fully in Phase 5):
  // { filename, iterations, quality_history, cleaning_plan, before_stats, after_stats, download_url }

  const score = results?.quality_score?.overall ?? 0
  const scoreColor = score >= 0.9 ? '#0F6E56' : score >= 0.7 ? '#854F0B' : '#A32D2D'
  const scoreBg   = score >= 0.9 ? '#E1F5EE' : score >= 0.7 ? '#FAEEDA' : '#FCEBEB'

  return (
    <div style={{ maxWidth: 860, margin: '0 auto', padding: '3rem 1.5rem' }}>

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '2rem' }}>
        <div>
          <p style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.12em', textTransform: 'uppercase', color: '#9a9a94', marginBottom: 4 }}>
            Pipeline complete
          </p>
          <h1 style={{ fontSize: 22, fontWeight: 600, color: '#1a1a18' }}>
            {results?.filename ?? 'Results'}
          </h1>
        </div>
        <button
          onClick={onReset}
          style={{ padding: '8px 16px', borderRadius: 8, fontSize: 13, border: '0.5px solid #d3d1c7', background: '#fff', cursor: 'pointer', color: '#5a5a56' }}
        >
          ← Clean another file
        </button>
      </div>

      {/* Metric cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 10, marginBottom: '2rem' }}>
        {[
          { label: 'Quality score', value: `${Math.round((results?.quality_score?.overall ?? 0) * 100)}%`, color: scoreColor, bg: scoreBg },
          { label: 'Iterations', value: results?.iteration ?? '—', color: '#185FA5', bg: '#E6F1FB' },
          { label: 'Columns cleaned', value: results?.cleaning_plan?.length ?? '—', color: '#534AB7', bg: '#EEEDFE' },
          { label: 'Missing values', value: `${Math.round((results?.quality_score?.missing_pct ?? 0) * 100)}%`, color: '#854F0B', bg: '#FAEEDA' },
        ].map(card => (
          <div key={card.label} style={{ padding: '14px 16px', background: '#fff', border: '0.5px solid #e5e3db', borderRadius: 10 }}>
            <p style={{ fontSize: 11, color: '#9a9a94', marginBottom: 6 }}>{card.label}</p>
            <p style={{ fontSize: 22, fontWeight: 600, color: card.color, background: card.bg, display: 'inline-block', padding: '2px 10px', borderRadius: 99 }}>
              {card.value}
            </p>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '0.5px solid #e5e3db', paddingBottom: 0 }}>
        {[['overview', 'Overview'], ['reasoning', 'Agent Reasoning'], ['history', 'Iteration History']].map(([key, label]) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            style={{
              padding: '8px 14px', fontSize: 13, fontWeight: tab === key ? 500 : 400,
              border: 'none', background: 'none', cursor: 'pointer',
              color: tab === key ? '#1a1a18' : '#9a9a94',
              borderBottom: tab === key ? '2px solid #1a1a18' : '2px solid transparent',
              marginBottom: -1,
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Tab: Overview */}
      {tab === 'overview' && (
        <div>
          <p style={{ fontSize: 13, color: '#5a5a56', marginBottom: 16 }}>
            Before/after column stats will display here once the pipeline is fully implemented (Phase 5).
          </p>
          {results?.download_url && (
            <a
              href={results.download_url}
              download
              style={{
                display: 'inline-block', padding: '10px 20px', background: '#1a1a18',
                color: '#fff', borderRadius: 8, fontSize: 13, fontWeight: 500, textDecoration: 'none',
              }}
            >
              Download cleaned CSV ↓
            </a>
          )}
        </div>
      )}

      {/* Tab: Agent Reasoning */}
      {tab === 'reasoning' && (
        <div>
          {results?.cleaning_plan?.length > 0 ? (
            results.cleaning_plan.map((step, i) => (
              <div key={i} style={{ padding: '14px 16px', background: '#fff', border: '0.5px solid #e5e3db', borderRadius: 10, marginBottom: 8 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                  <span style={{ fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 99, background: '#EEEDFE', color: '#534AB7' }}>
                    {step.column}
                  </span>
                  <span style={{ fontSize: 11, padding: '2px 8px', borderRadius: 99, background: '#E6F1FB', color: '#185FA5' }}>
                    {step.strategy} → {step.method}
                  </span>
                </div>
                <p style={{ fontSize: 13, color: '#5a5a56', lineHeight: 1.6 }}>{step.rationale}</p>
              </div>
            ))
          ) : (
            <p style={{ fontSize: 13, color: '#9a9a94' }}>
              Agent reasoning will appear here once the Decision Agent is implemented (Phase 3).
            </p>
          )}
        </div>
      )}

      {/* Tab: Iteration History */}
      {tab === 'history' && (
        <div>
          {results?.quality_history?.length > 0 ? (
            results.quality_history.map((q, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '12px 16px', background: '#fff', border: '0.5px solid #e5e3db', borderRadius: 10, marginBottom: 8 }}>
                <span style={{ fontSize: 12, fontWeight: 600, color: '#9a9a94', minWidth: 70 }}>Iteration {i + 1}</span>
                <div style={{ flex: 1, background: '#f5f4f0', borderRadius: 99, height: 6 }}>
                  <div style={{ width: `${q.overall * 100}%`, background: '#1D9E75', height: 6, borderRadius: 99, transition: 'width 0.5s' }} />
                </div>
                <span style={{ fontSize: 13, fontWeight: 500, color: '#1a1a18', minWidth: 40, textAlign: 'right' }}>
                  {Math.round(q.overall * 100)}%
                </span>
              </div>
            ))
          ) : (
            <p style={{ fontSize: 13, color: '#9a9a94' }}>
              Quality score history will appear here once the Validation Agent is implemented (Phase 4).
            </p>
          )}
        </div>
      )}

    </div>
  )
}
