"""
backend/main.py

FastAPI application — Phase 6
Exposes four endpoints:

  POST /api/upload          Accepts a CSV file, stores it in a temp session,
                            returns a session_id.

  POST /api/run             Runs the full LangGraph pipeline.  Streams
                            Server-Sent Events (SSE) so the frontend can show
                            live progress without polling.

  GET  /api/results/{run_id}  Returns the explainability.json payload for a
                            completed run (used when the frontend reloads).

  GET  /api/download/{run_id}/{file_type}
                            Streams the cleaned CSV or audit report for
                            download.  file_type ∈ {"csv", "report"}.

CORS is configured for local dev (Vite on :5173) and the production Vercel
origin.  Both origins are read from environment variables so the values never
have to be hardcoded.

Design decisions
----------------
- DataFrames are never stored in memory between requests; the CSV bytes are
  written to a per-session temp file on upload and re-read on /run.
- The pipeline is executed in a thread pool (run_in_executor) so it never
  blocks the event loop.
- SSE events are newline-delimited JSON objects.  The frontend can parse them
  with a simple EventSource.
- API keys are passed per-request in the request body — never stored on the
  server.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncGenerator

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

# Pipeline import — graph.py builds and compiles the LangGraph graph
from pipeline.graph import pipeline
from pipeline.state import PipelineState

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).parent
_TEMP_DIR    = _BACKEND_DIR / "data" / "tmp"
_OUTPUT_ROOT = _BACKEND_DIR / "data" / "output"
_TEMP_DIR.mkdir(parents=True, exist_ok=True)
_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Thread pool for blocking pipeline execution
# ---------------------------------------------------------------------------
_executor = ThreadPoolExecutor(max_workers=4)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Agentic Data Cleaner",
    description="Multi-agent LangGraph pipeline for context-aware CSV cleaning.",
    version="1.0.0",
)

# CORS — allow Vite dev server and production Vercel deployment
_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    os.getenv("FRONTEND_ORIGIN", "https://agentic-data-cleaner.vercel.app"),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    session_id: str = Field(..., description="ID returned by /api/upload")
    api_key: str    = Field(default="", description="Google Gemini API key (optional; omit for rules mode)")
    mode: str       = Field(default="rules", description="'llm' or 'rules'")
    max_iterations: int   = Field(default=3, ge=1, le=5)
    quality_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    column_overrides: dict = Field(default={}, description="User overrides: {col: action} e.g. {'Age': 'drop_column'}")


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    rows: int
    columns: int
    column_names: list[str]


class RunResponse(BaseModel):
    """Returned as the final SSE event payload."""
    run_id: str
    overall_score: float
    iterations_completed: int
    shape_before: list[int]
    shape_after: list[int]
    explainability_url: str
    csv_download_url: str
    report_download_url: str


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse(event: str, data: dict) -> str:
    """Format a single Server-Sent Events message."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def _stream_pipeline(
    session_id: str,
    request: RunRequest,
) -> AsyncGenerator[str, None]:
    """
    Async generator that runs the pipeline in a thread and yields SSE events.

    Events emitted (in order):
      started   — pipeline has begun
      progress  — one event per completed agent node
      done      — pipeline finished, payload = RunResponse dict
      error     — pipeline failed, payload = {"detail": str}
    """
    loop = asyncio.get_event_loop()

    # Resolve the uploaded file — saved as {session_id}__{filename}.csv
    matches = list(_TEMP_DIR.glob(f"{session_id}__*.csv"))
    if not matches:
        yield _sse("error", {"detail": f"Session '{session_id}' not found or expired. Please re-upload the file."})
        return
    tmp_path = matches[0]

    yield _sse("started", {"session_id": session_id, "mode": request.mode})

    try:
        # Load CSV from temp file
        df = pd.read_csv(tmp_path)
        filename = tmp_path.name.split("__", 1)[-1]

        initial_state: PipelineState = {
            "raw_dataset": df,
            "current_dataset": None,
            "filename": filename,
            "profile": None,
            "cleaning_plan": None,
            "quality_score": None,
            "iteration": 0,
            "max_iterations": request.max_iterations,
            "quality_threshold": request.quality_threshold,
            "audit_log": [],
            "quality_history": [],
            "api_key": request.api_key,
            "mode": request.mode if request.api_key else "rules",
            "column_overrides": request.column_overrides,
            "output_paths": None,
            "explainability": None,
        }

        # Emit a progress event for each pipeline stage as it completes.
        # LangGraph's stream() yields (node_name, output_state) tuples.
        final_state: PipelineState | None = None

        node_label = {
            "ingestion":  "Ingesting and validating data…",
            "profiling":  "Profiling columns…",
            "decision":   "Building cleaning plan…",
            "execution":  "Applying transformations…",
            "validation": "Scoring data quality…",
            "output":     "Writing artefacts…",
        }

        def run_stream():
            """
            Stream the pipeline, accumulating ALL node outputs into one state dict.
            LangGraph 0.1.x yields {node_name: partial_output} per node —
            we merge every partial into final_state so no keys are lost.
            Returns (node_names_in_order, accumulated_state).
            """
            accumulated = dict(initial_state)  # start with full initial state
            node_names = []
            for chunk in pipeline.stream(initial_state):
                if isinstance(chunk, dict):
                    for node_name, node_output in chunk.items():
                        node_names.append(node_name)
                        if isinstance(node_output, dict):
                            accumulated.update(node_output)
            return node_names, accumulated

        node_names, final_state = await loop.run_in_executor(_executor, run_stream)

        for node_name in node_names:
            label = node_label.get(node_name, node_name)
            yield _sse("progress", {"node": node_name, "label": label})

        if final_state is None or not final_state.get("output_paths"):
            yield _sse("error", {"detail": "Pipeline completed but produced no output."})
            return

        output_paths = final_state["output_paths"]
        run_id       = output_paths["run_id"]
        explain      = final_state.get("explainability") or {}
        qs           = final_state.get("quality_score") or {}

        shape_before = explain.get("shape_before", [0, 0])
        shape_after  = explain.get("shape_after",  [0, 0])

        payload = RunResponse(
            run_id=run_id,
            overall_score=qs.get("overall", 0.0),
            iterations_completed=final_state.get("iteration", 0),
            shape_before=shape_before,
            shape_after=shape_after,
            explainability_url=f"/api/results/{run_id}",
            csv_download_url=f"/api/download/{run_id}/csv",
            report_download_url=f"/api/download/{run_id}/report",
        )
        yield _sse("done", payload.model_dump())

    except Exception as exc:
        logger.exception("Pipeline failed for session %s", session_id)
        yield _sse("error", {"detail": str(exc)})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/upload", response_model=UploadResponse, summary="Upload a CSV file")
async def upload_csv(file: UploadFile = File(...)) -> UploadResponse:
    """
    Accept a CSV upload, store it temporarily, return metadata.

    The session_id must be passed to /api/run to identify the file.
    Files are stored at backend/data/tmp/<session_id>__<filename>.csv.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > 50 * 1024 * 1024:   # 50 MB hard limit
        raise HTTPException(status_code=413, detail="File exceeds 50 MB limit.")

    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {exc}") from exc

    if df.empty:
        raise HTTPException(status_code=422, detail="CSV file contains no data rows.")
    if df.shape[1] < 2:
        raise HTTPException(status_code=422, detail="CSV must have at least 2 columns.")

    # Persist to temp dir
    session_id = str(uuid.uuid4())
    safe_name  = "".join(c if c.isalnum() or c in "-_." else "_" for c in file.filename)
    tmp_path   = _TEMP_DIR / f"{session_id}__{safe_name}"
    tmp_path.write_bytes(contents)
    logger.info("Upload saved → %s  (%d rows × %d cols)", tmp_path.name, *df.shape)

    return UploadResponse(
        session_id=session_id,
        filename=file.filename,
        rows=df.shape[0],
        columns=df.shape[1],
        column_names=list(df.columns),
    )


@app.post("/api/run", summary="Run the cleaning pipeline (SSE stream)")
async def run_pipeline(request: RunRequest) -> StreamingResponse:
    """
    Stream pipeline progress as Server-Sent Events.

    The client should open this with EventSource or fetch + ReadableStream.
    Each message is: `event: <name>\\ndata: <json>\\n\\n`

    Events:  started | progress | done | error
    """
    return StreamingResponse(
        _stream_pipeline(request.session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable Nginx buffering
        },
    )


@app.get("/api/results/{run_id}", summary="Get explainability payload for a run")
async def get_results(run_id: str) -> dict:
    """
    Return the explainability.json for a completed run.
    Used by ResultsPage on initial load and on page refresh.
    """
    json_path = _OUTPUT_ROOT / run_id / "explainability.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return json.loads(json_path.read_text(encoding="utf-8"))


@app.get("/api/download/{run_id}/{file_type}", summary="Download a run artefact")
async def download_artefact(run_id: str, file_type: str) -> FileResponse:
    """
    Stream a run artefact for download.

    file_type:
      csv     → cleaned_<stem>.csv
      report  → audit_report.md
    """
    run_dir = _OUTPUT_ROOT / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    if file_type == "csv":
        matches = list(run_dir.glob("cleaned_*.csv"))
        if not matches:
            raise HTTPException(status_code=404, detail="CSV artefact not found.")
        file_path = matches[0]
        media_type = "text/csv"
    elif file_type == "csv-dedup":
        matches = list(run_dir.glob("cleaned_*.csv"))
        if not matches:
            raise HTTPException(status_code=404, detail="CSV artefact not found.")
        import tempfile
        df_dd = pd.read_csv(matches[0])
        before = len(df_dd)
        df_dd = df_dd.drop_duplicates()
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode='w')
        df_dd.to_csv(tmp.name, index=False)
        tmp.close()
        logger.info("Dedup download: %d → %d rows", before, len(df_dd))
        return FileResponse(path=tmp.name, media_type="text/csv", filename=f"deduped_{matches[0].name}")
    elif file_type == "report":
        file_path = run_dir / "audit_report.md"
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Report artefact not found.")
        media_type = "text/markdown"
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown file_type '{file_type}'. Use 'csv' or 'report'.",
        )

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=file_path.name,
    )


@app.post("/api/summary/{run_id}", summary="Generate AI narrative summary for a run")
async def generate_summary(run_id: str, body: dict = None) -> dict:
    """
    Use Gemini to generate a plain-English dataset summary paragraph.
    Requires api_key in request body: {"api_key": "AIza..."}
    Falls back to a rule-based summary if no key is provided.
    """
    json_path = _OUTPUT_ROOT / run_id / "explainability.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    api_key = (body or {}).get("api_key", "")

    profiles = data.get("column_profiles", [])
    plan     = data.get("cleaning_plan", [])
    score    = data.get("final_score", {})
    shape_b  = data.get("shape_before", [0, 0])
    shape_a  = data.get("shape_after", [0, 0])
    filename = data.get("filename", "dataset")
    iters    = data.get("iterations_completed", 1)

    # Build a compact context string for the LLM
    high_missing = [p["column"] for p in profiles if (p.get("missing_pct") or 0) > 0.2]
    high_skew    = [p["column"] for p in profiles if (p.get("skewness") or 0) and abs(p["skewness"]) > 1.5]
    high_outlier = [p["column"] for p in profiles if (p.get("outlier_ratio") or 0) > 0.05]
    numeric_cols = [p["column"] for p in profiles if p.get("skewness") is not None]
    cat_cols     = [p["column"] for p in profiles if p.get("skewness") is None]
    actions      = {}
    for s in plan:
        actions.setdefault(s["action"], []).append(s["column"])

    # Rule-based fallback summary (always computed)
    fallback = (
        f"{filename} has {shape_b[0]:,} rows and {shape_b[1]} columns "
        f"({len(numeric_cols)} numeric, {len(cat_cols)} categorical). "
        f"After {iters} cleaning iteration(s), the overall data quality score reached "
        f"{round(score.get('overall', 0) * 100)}% "
        f"(missing: {round(score.get('missing_score', 0) * 100)}%, "
        f"outlier: {round(score.get('outlier_score', 0) * 100)}%, "
        f"skewness: {round(score.get('skewness_score', 0) * 100)}%). "
    )
    if high_missing:
        fallback += f"Columns with significant missing data: {', '.join(high_missing[:4])}. "
    if high_skew:
        fallback += f"Right-skewed distributions corrected: {', '.join(high_skew[:4])}. "
    if "drop_column" in actions:
        fallback += f"Dropped entirely-null columns: {', '.join(actions['drop_column'])}. "
    fallback += f"The cleaned dataset has {shape_a[0]:,} rows and {shape_a[1]} columns, ready for ML."

    if not api_key:
        return {"summary": fallback, "source": "rules"}

    # LLM-based summary
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0.3,
            max_output_tokens=300,
        )

        system = (
            "You are a senior data scientist writing a brief, plain-English summary of a dataset "
            "cleaning run for a non-technical stakeholder. Write exactly 3-4 sentences. "
            "Be specific — mention column names, percentages, and what was done. "
            "Do NOT use bullet points or headers. Output only the paragraph, nothing else."
        )

        numeric_str  = ", ".join(numeric_cols[:8]) or "none"
        cat_str      = ", ".join(cat_cols[:8]) or "none"
        missing_str  = ", ".join(high_missing) or "none"
        skew_str     = ", ".join(high_skew) or "none"
        outlier_str  = ", ".join(high_outlier) or "none"
        actions_str  = str({k: v for k, v in list(actions.items())[:6]})
        score_str    = (
            f"{round(score.get('overall', 0) * 100)}% "
            f"(missing {round(score.get('missing_score', 0)*100)}%, "
            f"outlier {round(score.get('outlier_score', 0)*100)}%, "
            f"skewness {round(score.get('skewness_score', 0)*100)}%)"
        )
        user = (
            f"Dataset: {filename}\n"
            f"Shape: {shape_b[0]} rows x {shape_b[1]} cols -> {shape_a[0]} rows x {shape_a[1]} cols\n"
            f"Numeric columns: {numeric_str}\n"
            f"Categorical columns: {cat_str}\n"
            f"High missing (>20%): {missing_str}\n"
            f"High skew (>1.5): {skew_str}\n"
            f"High outliers (>5%): {outlier_str}\n"
            f"Actions taken: {actions_str}\n"
            f"Final quality score: {score_str}\n"
            f"Iterations: {iters}\n\n"
            "Write 3-4 sentences summarising what this dataset is, "
            "what quality issues were found, and what was fixed."
        )

        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return {"summary": response.content.strip(), "source": "llm"}

    except Exception as exc:
        logger.warning("AI summary failed (%s) — falling back to rules.", exc)
        return {"summary": fallback, "source": "rules"}


@app.get("/api/download/{run_id}/parquet", summary="Download cleaned data as Parquet")
async def download_parquet(run_id: str):
    run_dir = _OUTPUT_ROOT / run_id
    matches = list(run_dir.glob("cleaned_*.csv"))
    if not matches:
        raise HTTPException(status_code=404, detail="CSV artefact not found.")
    df = pd.read_csv(matches[0])
    import io as _io
    buf = _io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    from fastapi.responses import Response
    stem = matches[0].stem
    return Response(
        content=buf.read(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={stem}.parquet"}
    )


@app.get("/api/health", summary="Health check")
async def health() -> dict:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)