# Agentic Data Cleaner

A full-stack AI web application that cleans messy CSV files using a
**multi-agent LangGraph pipeline** where a Decision Agent (Gemini) reasons
context-aware transformations with per-column rationale, and a Validation
Agent drives iterative quality improvement.

**Live demo:** 
**API docs:** 

---

## Why this project

Most data cleaning tools apply static rules blindly. This project is different:

- **Explainability** — every transformation has a rationale written by the LLM, not a terse code label
- **Iterative agentic reasoning** — the pipeline loops up to 3× until a quality threshold is met
- **Human-in-the-loop** — users can override any agent decision before the pipeline runs
- **Two modes** — LLM (Gemini reasons the plan) with automatic fallback to a deterministic rules engine
- **Full audit trail** — Markdown report + explainability JSON document every decision, iteration, and quality score
- **Rich analytics** — correlation heatmap, before/after distributions, ML readiness scoring, data lineage graph
- **Run history** — all past runs saved locally; re-open, re-download, or share any run via URL

---

## What it produces

For every CSV upload the pipeline outputs:

| Artefact | Format | Contents |
|---|---|---|
| Cleaned dataset | `.csv` | Transformed, imputed, encoded data |
| Cleaned dataset | `.parquet` | Same data in columnar format for analytics tools |
| Audit report | `.md` | Per-column decisions, quality scores, execution log |
| Explainability | `.json` | Full structured payload consumed by the UI |

---

## Architecture

```
Upload CSV
    │
    ▼
┌─────────────┐   FastAPI /api/run (SSE stream)
│  Ingestion  │   dtype coercion, empty-row removal
└──────┬──────┘
       │
┌──────▼──────┐
│  Profiling  │   missing_pct, outlier_ratio, skewness, histogram bins,
│             │   value_counts, cardinality, numeric_summary (Q1/Q3/median)
└──────┬──────┘
       │
┌──────▼──────┐
│  Decision   │   Gemini (LLM) or rules engine → CleaningPlan
│             │   User column_overrides applied on top
└──────┬──────┘
       │
┌──────▼──────┐
│  Execution  │   16 action handlers (impute, encode, scale, transform…)
└──────┬──────┘
       │
┌──────▼──────┐
│ Validation  │   QualityScore = 0.5×missing + 0.3×outlier + 0.2×skewness
└──────┬──────┘
       │
  score ≥ threshold        score < threshold
  OR max iterations        AND iterations remain
       │                         │
       │                    loop back to Decision
       ▼
┌──────────────┐
│    Output    │   cleaned CSV + Parquet + audit_report.md +
│              │   explainability.json (with correlation matrix,
│              │   before/after profiles, duplicate count)
└──────────────┘
```

---

## Results dashboard

After cleaning, the UI shows:

| Tab | What you get |
|---|---|
| Overview | Shape before/after, quality sub-scores, column profiles table, AI dataset summary, downloads |
| 📊 Data Viewer | Browse full cleaned CSV with column toggle, row search, pagination, filtered download |
| 📉 Distributions | Before/after histogram overlay per numeric column; value-frequency bars for categoricals |
| ⚠ Alerts | Ranked data quality issues (critical / warning / info) each linked to the agent's fix |
| 🔍 Column Insights | Per-column card with missing bar, outlier bar, skewness gauge, action badge, ML readiness |
| 🔗 Correlation | Pearson heatmap with hover tooltip; top-6 strongest pairs labelled Weak / Moderate / Strong |
| 🎯 ML Readiness | Columns grouped into Ready / Review / High Risk / Dropped with per-issue tags |
| 🔀 Data Lineage | Column → transformation arrow → output, colour-coded by action category |
| Agent Reasoning | Grouped by action category (imputation, encoding, scaling…) with per-column rationale |
| Iteration History | Quality score progression per iteration with missing / outlier / skewness breakdown |

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React 18 + Vite |
| Backend | FastAPI + Uvicorn |
| Orchestration | LangGraph 0.1 |
| LLM | Gemini 2.0 Flash via `langchain-google-genai` |
| Data | Pandas, NumPy |
| ML utils | Scikit-learn |
| Testing | pytest + httpx + pytest-asyncio |
| Deployment | Render (backend) + Vercel (frontend) |

---

## Project structure

```
agentic-data-cleaner/
├── backend/
│   ├── main.py                  FastAPI app — all endpoints
│   ├── requirements.txt
│   ├── .env.example
│   ├── agents/
│   │   ├── ingestion.py         dtype coercion, validation
│   │   ├── profiling_agent.py   per-column stats + histograms + value_counts
│   │   ├── decision_agent.py    LLM + rules engine + user overrides
│   │   ├── execution_agent.py   16 transformation handlers
│   │   ├── validation_agent.py  QualityScore, drives loop
│   │   └── output.py            CSV + Parquet + Markdown + explainability JSON
│   ├── pipeline/
│   │   ├── state.py             PipelineState TypedDict
│   │   └── graph.py             LangGraph graph + conditional edges
│   └── data/
│       ├── sample/              titanic.csv, house_prices.csv
│       ├── tmp/                 uploaded files (ephemeral)
│       └── output/              per-run artefacts
├── frontend/
│   ├── src/
│   │   ├── App.jsx              routing: upload / results / history / shared report
│   │   ├── pages/
│   │   │   ├── UploadPage.jsx   drag-drop + mode selector + column overrides editor
│   │   │   ├── ResultsPage.jsx  10-tab results dashboard
│   │   │   ├── HistoryPage.jsx  localStorage run history
│   │   │   └── SharedReportPage.jsx  public report viewer (/report/:run_id)
│   │   └── api/client.js        axios + SSE stream wrappers
│   ├── package.json
│   ├── vite.config.js
│   └── index.html
├── tests/
│   ├── test_pipeline.py
│   ├── test_decision_agent.py
│   ├── test_phase4.py
│   ├── test_output.py
│   └── test_api.py
├── render.yaml
├── vercel.json
├── Makefile
└── README.md
```

---

## Quick start

### Prerequisites

- Python 3.11+
- Node.js 18+
- A Gemini API key — [get one free](https://aistudio.google.com/app/apikey) (optional — rules mode works without one)

### 1. Clone and install

```bash
git clone https://github.com/your-username/agentic-data-cleaner.git
cd agentic-data-cleaner
make install
```

### 2. Configure environment

```bash
cp backend/.env.example backend/.env
# Set GEMINI_API_KEY=AIzaSy... in backend/.env (optional)
```

### 3. Run locally

```bash
# Terminal 1
make dev-backend      # uvicorn on :8000

# Terminal 2
make dev-frontend     # vite on :5173
```

Open http://localhost:5173.

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/upload` | Upload CSV → `session_id` |
| `POST` | `/api/run` | Run pipeline (SSE stream) |
| `GET` | `/api/results/{run_id}` | Full explainability JSON |
| `GET` | `/api/download/{run_id}/csv` | Cleaned CSV |
| `GET` | `/api/download/{run_id}/csv-dedup` | Deduplicated CSV |
| `GET` | `/api/download/{run_id}/parquet` | Cleaned data as Parquet |
| `GET` | `/api/download/{run_id}/report` | Markdown audit report |
| `POST` | `/api/summary/{run_id}` | AI narrative summary (pass `{"api_key": "..."}`) |
| `GET` | `/api/health` | Health check |

Interactive docs: http://localhost:8000/docs

### RunRequest body

```json
{
  "session_id": "...",
  "api_key": "AIzaSy...",
  "mode": "llm",
  "max_iterations": 3,
  "quality_threshold": 0.90,
  "column_overrides": {
    "Age": "drop_column",
    "Cabin": "keep"
  }
}
```

`column_overrides` is optional. Any action from the supported list can be specified per column — the agent's decision is replaced with the user's choice.

### SSE event types

```
event: started    data: {"session_id": "...", "mode": "llm"}
event: progress   data: {"node": "profiling", "label": "Profiling columns…"}
event: done       data: {"run_id": "...", "overall_score": 0.94, ...}
event: error      data: {"detail": "..."}
```

---

## Supported cleaning actions

| Category | Actions |
|---|---|
| Missing values | `impute_mean`, `impute_median`, `impute_mode`, `impute_constant`, `drop_column` |
| Outliers | `clip_iqr`, `winsorise` |
| Encoding | `encode_onehot`, `encode_ordinal`, `encode_binary` |
| Scaling | `scale_standard`, `scale_minmax`, `scale_robust` |
| Transforms | `log_transform`, `sqrt_transform` |
| No-op | `keep` |

---

## Running tests

```bash
make test                        # all tests
pytest tests/test_pipeline.py    # ingestion + profiling
pytest tests/test_decision_agent.py
pytest tests/test_phase4.py      # execution + validation + graph routing
pytest tests/test_output.py
pytest tests/test_api.py         # requires pytest-asyncio
```

---

## Sharing reports

Every completed run generates a shareable URL:

```
https://your-app.vercel.app/report/{run_id}
```

Anyone with the link can view the full results dashboard — no login required. The run data lives on the Render backend for as long as the server's ephemeral storage persists (redeploy clears it; attach a persistent disk on Render for durability).

## License

MIT