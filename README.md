# Agentic Data Cleaner

A full-stack AI web application that cleans messy CSV files using a
**multi-agent LangGraph pipeline** where a Decision Agent (Claude) reasons
context-aware transformations with per-column rationale, and a Validation
Agent drives iterative quality improvement.

**Live demo:** https://agentic-data-cleaner.vercel.app  
**API:** https://agentic-data-cleaner-api.onrender.com/docs

---

## Why this project

Most data cleaning tools apply static rules blindly.  This project is different:

- **Explainability** — every transformation comes with a rationale written by Claude, not a terse code label
- **Iterative agentic reasoning** — the pipeline loops up to 3× until a quality threshold is met
- **Two modes** — LLM (Claude reasons the plan) with automatic fallback to a deterministic rules engine
- **Full audit trail** — a Markdown report documents every decision, iteration, and quality score

---

## Architecture

```
Upload CSV
    │
    ▼
┌─────────────┐   FastAPI /api/run (SSE stream)
│  Ingestion  │  dtype coercion, empty-row removal
└──────┬──────┘
       │
┌──────▼──────┐
│  Profiling  │  missing_pct, outlier_ratio, skewness, cardinality
└──────┬──────┘
       │
┌──────▼──────┐
│  Decision   │  Claude (LLM) or rules engine → CleaningPlan
└──────┬──────┘
       │
┌──────▼──────┐
│  Execution  │  applies 16 actions (impute, encode, scale, transform…)
└──────┬──────┘
       │
┌──────▼──────┐
│ Validation  │  QualityScore = 0.5×missing + 0.3×outlier + 0.2×skewness
└──────┬──────┘
       │
  score ≥ threshold        score < threshold
  OR max iterations        AND iterations remain
       │                         │
       │                    loop back to Decision
       ▼
┌──────────────┐
│    Output    │  cleaned CSV + audit_report.md + explainability.json
└──────────────┘
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React 18 + Vite |
| Backend | FastAPI + Uvicorn |
| Orchestration | LangGraph |
| LLM | Claude (`claude-opus-4-5`) via `langchain-anthropic` |
| Data | Pandas, NumPy |
| ML utils | Scikit-learn |
| Testing | pytest + httpx |
| Deployment | Render (backend) + Vercel (frontend) |

---

## Project structure

```
agentic-data-cleaner/
├── backend/
│   ├── main.py                  FastAPI app — /upload, /run (SSE), /results, /download
│   ├── cli.py                   Command-line interface
│   ├── requirements.txt
│   ├── .env.example
│   ├── agents/
│   │   ├── ingestion.py         dtype coercion, validation
│   │   ├── profiling_agent.py   per-column statistics
│   │   ├── decision_agent.py    LLM + rules engine → CleaningPlan
│   │   ├── execution_agent.py   applies 16 transformation handlers
│   │   ├── validation_agent.py  QualityScore, drives loop
│   │   └── output.py            CSV + Markdown report + explainability JSON
│   ├── pipeline/
│   │   ├── state.py             PipelineState TypedDict
│   │   └── graph.py             LangGraph graph + conditional edges
│   └── data/
│       ├── sample/              titanic.csv, house_prices.csv
│       ├── tmp/                 uploaded files (ephemeral)
│       └── output/              per-run artefacts
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── pages/
│   │   │   ├── UploadPage.jsx   drag-drop + API key + run button
│   │   │   └── ResultsPage.jsx  quality cards + reasoning tab + history tab
│   │   └── api/client.js        axios + EventSource wrappers
│   ├── package.json
│   ├── vite.config.js
│   └── index.html
├── tests/
│   ├── test_pipeline.py         Phase 2: ingestion + profiling
│   ├── test_decision_agent.py   Phase 3: rules engine + mocked LLM
│   ├── test_phase4.py           Phase 4: execution + validation + graph routing
│   ├── test_output.py           Phase 5: report builders + run_output node
│   └── test_api.py              Phase 6: FastAPI endpoints
├── render.yaml                  Render deployment config
├── vercel.json                  Vercel deployment config
├── Makefile
└── README.md
```

---

## Quick start

### Prerequisites

- Python 3.11+
- Node.js 18+
- An Anthropic API key (optional — rules mode works without one)

### 1. Clone and install

```bash
git clone https://github.com/your-username/agentic-data-cleaner.git
cd agentic-data-cleaner
make install          # pip install -r backend/requirements.txt && npm install (frontend)
```

### 2. Configure environment

```bash
cp backend/.env.example backend/.env
# Edit backend/.env and set ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run locally

```bash
# Terminal 1 — backend
make dev-backend      # uvicorn main:app --reload on :8000

# Terminal 2 — frontend
make dev-frontend     # vite dev server on :5173
```

Open http://localhost:5173 in your browser.

---

## CLI usage

```bash
# Rules mode (no API key required):
python backend/cli.py backend/data/sample/titanic.csv

# LLM mode:
python backend/cli.py backend/data/sample/titanic.csv \
    --mode llm --api-key $ANTHROPIC_API_KEY

# Custom thresholds:
python backend/cli.py backend/data/sample/house_prices.csv \
    --mode llm --api-key $ANTHROPIC_API_KEY \
    --max-iterations 5 --threshold 0.95

# Quiet (JSON output only — useful for scripting):
python backend/cli.py data.csv --quiet | jq .overall_score
```

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/upload` | Upload CSV → returns `session_id` |
| `POST` | `/api/run` | Run pipeline (SSE stream of progress events) |
| `GET` | `/api/results/{run_id}` | Fetch explainability JSON |
| `GET` | `/api/download/{run_id}/csv` | Download cleaned CSV |
| `GET` | `/api/download/{run_id}/report` | Download Markdown audit report |
| `GET` | `/api/health` | Health check |

Interactive docs: `http://localhost:8000/docs`

### SSE event types

```
event: started    data: {"session_id": "...", "mode": "llm"}
event: progress   data: {"node": "profiling", "label": "Profiling columns…"}
event: done       data: {"run_id": "...", "overall_score": 0.94, ...}
event: error      data: {"detail": "..."}
```

---

## Running tests

```bash
# All tests
make test                       # pytest tests/ -v

# Individual suites
pytest tests/test_pipeline.py   # Phase 2
pytest tests/test_decision_agent.py
pytest tests/test_phase4.py
pytest tests/test_output.py
pytest tests/test_api.py        # requires: pip install httpx pytest-asyncio
```

---

## Deploying

### Backend → Render

1. Push to GitHub.
2. Render → **New → Blueprint** → connect repo.
3. Render auto-detects `render.yaml`.
4. Set `ANTHROPIC_API_KEY` in the Render dashboard under **Environment**.

### Frontend → Vercel

1. Vercel → **New Project** → import repo.
2. Set **Root Directory** to `frontend`.
3. Vercel auto-detects Vite.
4. Update the `/api` rewrite target in `vercel.json` to your Render URL.

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



## License

MIT
