# Project Handoff: Agentic Data Cleaner

Paste this entire file at the start of a new Claude chat to continue building.

---

## What We Are Building

A full-stack AI web application called **Agentic Data Cleaner**.

Users upload a messy CSV file on a React frontend. A FastAPI backend runs a
LangGraph multi-agent pipeline that:
1. Profiles the dataset (missing values, outliers, skewness, dtypes)
2. Uses Claude (LLM) to reason a context-aware cleaning plan with a rationale per column
3. Executes the transformations (imputation, encoding, scaling, outlier handling)
4. Validates data quality and loops back if the threshold isn't met (max 3 iterations)
5. Exports the cleaned CSV + a Markdown audit report

This is a resume/portfolio project. The key differentiator is **explainability + iterative
agentic reasoning** — not just static rules.

---

## Tech Stack (all free)

| Layer          | Technology                        |
|----------------|-----------------------------------|
| Frontend       | React 18 + Vite                   |
| Backend        | FastAPI + Uvicorn                 |
| Orchestration  | LangGraph                         |
| LLM            | Claude via langchain-anthropic    |
| Data           | Pandas, NumPy                     |
| ML utils       | Scikit-learn                      |
| Output format  | CSV + PyArrow/Parquet             |
| Testing        | pytest                            |
| Deployment     | Render (backend) + Vercel (frontend) — both free |

---

## Folder Structure (already created)

```
agentic-data-cleaner/
├── backend/
│   ├── main.py                  ✅ FastAPI entry point (stub /upload and /run endpoints)
│   ├── requirements.txt         ✅ All dependencies pinned
│   ├── .env.example             ✅
│   ├── agents/
│   │   ├── ingestion.py         ⬜ TODO Phase 2
│   │   ├── profiling_agent.py   ⬜ TODO Phase 2
│   │   ├── decision_agent.py    ⬜ TODO Phase 3
│   │   ├── execution_agent.py   ⬜ TODO Phase 4
│   │   ├── validation_agent.py  ⬜ TODO Phase 4
│   │   └── output.py            ⬜ TODO Phase 5
│   ├── pipeline/
│   │   ├── state.py             ✅ PipelineState TypedDict fully defined
│   │   └── graph.py             ✅ LangGraph graph + conditional edges wired
│   └── data/sample/             ⬜ Add titanic.csv and house_prices.csv here
├── frontend/
│   ├── src/
│   │   ├── App.jsx              ✅ Page routing (Upload ↔ Results)
│   │   ├── pages/
│   │   │   ├── UploadPage.jsx   ✅ Drag-drop upload + API key input + run button
│   │   │   └── ResultsPage.jsx  ✅ Quality cards + reasoning tab + iteration history tab
│   │   └── api/client.js        ✅ axios calls to FastAPI
│   ├── package.json             ✅
│   ├── vite.config.js           ✅ Proxy /api → localhost:8000
│   └── index.html               ✅
├── tests/
│   └── test_pipeline.py         ✅ Basic sanity tests (expand per phase)
├── .gitignore                   ✅
├── Makefile                     ✅ make install / make dev-backend / make dev-frontend
└── README.md                    ✅
```

---

## PipelineState (state.py) — key fields

```python
class PipelineState(TypedDict):
    raw_dataset: Optional[pd.DataFrame]
    current_dataset: Optional[pd.DataFrame]
    filename: str
    profile: Optional[List[ColumnProfile]]      # output of profiling agent
    cleaning_plan: Optional[List[CleaningStep]] # output of decision agent
    quality_score: Optional[QualityScore]       # output of validation agent
    iteration: int
    max_iterations: int                         # default 3
    quality_threshold: float                    # default 0.90
    audit_log: List[Dict]
    quality_history: List[QualityScore]
    api_key: str
    mode: str                                   # "llm" or "rules"
```

---

## LangGraph Graph (graph.py) — already wired

```
ingestion → profiling → decision → execution → validation
                           ↑_________________________↓  (if score < threshold AND iter < max)
                                        ↓  (if score >= threshold OR iter >= max)
                                     output → END
```

Conditional edge function `should_continue()` already implemented.

---

## 6-Phase Build Plan

| Phase | What to build                          | Timeline  | Status |
|-------|----------------------------------------|-----------|--------|
| 1     | Setup, folder structure, env           | Days 1–2  | ✅ Done |
| 2     | Ingestion + Profiling agents           | Days 3–4  | ⬜ Next |
| 3     | Decision Agent (LLM + rules fallback)  | Days 5–7  | ⬜      |
| 4     | Execution + Validation agents          | Days 8–10 | ⬜      |
| 5     | Output, audit report, explainability   | Days 11–12| ⬜      |
| 6     | CLI, wire FastAPI /run endpoint, deploy| Days 13–14| ⬜      |

---

## What To Do Next (Phase 2)

Implement these two files:

### backend/agents/ingestion.py
- Receive `state["raw_dataset"]` (already a DataFrame)
- Detect and fix dtype issues (e.g. numeric columns stored as strings)
- Validate: non-empty, has at least 2 columns, no completely empty rows
- Write validated df to `state["current_dataset"]`
- Append `{"step": "ingestion", "status": "ok", "shape": [...]}` to `state["audit_log"]`

### backend/agents/profiling_agent.py
- Iterate over all columns in `state["current_dataset"]`
- Per column compute:
  - `missing_pct`: null count / total rows
  - `unique_count`: number of unique values
  - `skewness`: for numeric columns only (pandas .skew())
  - `outlier_ratio`: IQR method — (Q1 - 1.5*IQR, Q3 + 1.5*IQR) for numeric cols
  - `dtype`: pandas dtype as string
  - `sample_values`: first 3 non-null values
- Write list of `ColumnProfile` dicts to `state["profile"]`
- Append to `state["audit_log"]`

---

## Open Datasets to Use

| Dataset          | URL                                 | License |
|------------------|-------------------------------------|---------|
| Titanic          | kaggle.com/c/titanic                | CC0     |
| House Prices     | kaggle.com/c/house-prices           | CC0     |
| UCI Heart Disease| archive.ics.uci.edu/dataset/45      | CC BY 4.0|

Download titanic.csv and house_prices.csv and place them in:
`backend/data/sample/`

---

## Resume Bullet (use this)

"Built and deployed a full-stack AI application — React frontend, FastAPI backend —
using LangGraph to orchestrate a multi-agent data cleaning pipeline where a Decision
Agent (Claude) reasons context-aware transformations with per-column rationale,
a Validation Agent drives iterative quality improvement, reducing missing values by
up to 100% and outlier ratio by ~60% across benchmark datasets."

---

## How To Start The New Chat

Paste this file and say:
"Let's continue building the Agentic Data Cleaner. We are starting Phase 2.
Please implement ingestion.py and profiling_agent.py."
