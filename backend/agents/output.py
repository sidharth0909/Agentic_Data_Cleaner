"""
backend/agents/output.py

Output Agent — Phase 5
The terminal node of the pipeline.  It receives the fully-cleaned state and
produces three artefacts:

  1. cleaned_<filename>.csv      — the cleaned DataFrame as UTF-8 CSV
  2. audit_report.md             — human-readable Markdown audit report
  3. explainability.json         — structured JSON consumed by the frontend
                                   Results page (Reasoning tab + iteration
                                   history tab)

All artefacts are written to a per-run output directory:
  backend/data/output/<run_id>/

The node writes paths back into state["output"] so the FastAPI /run endpoint
can stream them to the client.

Design notes
------------
- run_id is derived from filename + UTC timestamp so parallel runs never
  collide.
- The Markdown report is self-contained — a recruiter can read it without
  the frontend.
- explainability.json mirrors the shape the React ResultsPage expects:
    {
      "run_id": str,
      "filename": str,
      "final_score": QualityScore,
      "quality_history": [QualityScore, ...],
      "cleaning_plan": [CleaningStep, ...],      ← rationale per column
      "execution_log": [step_log_entry, ...],    ← from audit_log
      "column_profiles": [ColumnProfile, ...],   ← initial profile
      "iterations_completed": int,
      "shape_before": [rows, cols],
      "shape_after":  [rows, cols],
    }
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.state import (
    CleaningStep,
    ColumnProfile,
    PipelineState,
    QualityScore,
)

logger = logging.getLogger(__name__)

# Root directory for all pipeline outputs (relative to backend/)
_OUTPUT_ROOT = Path(__file__).parent.parent / "data" / "output"


# ---------------------------------------------------------------------------
# Run-ID helper
# ---------------------------------------------------------------------------

def _make_run_id(filename: str) -> str:
    """
    Deterministic, filesystem-safe run identifier.
    Format: <stem>_<YYYYmmdd_HHMMSS_UTC>
    Example: titanic_20240315_142301_UTC
    """
    stem = re.sub(r"[^\w]", "_", Path(filename).stem)[:40]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    return f"{stem}_{ts}"


# ---------------------------------------------------------------------------
# Artefact builders
# ---------------------------------------------------------------------------

def _build_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _score_bar(score: float, width: int = 20) -> str:
    """ASCII progress bar for the Markdown report."""
    filled = round(score * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {score * 100:.1f}%"


def _build_markdown_report(
    filename: str,
    run_id: str,
    shape_before: list[int],
    shape_after: list[int],
    initial_profile: list[ColumnProfile],
    cleaning_plan: list[CleaningStep],
    quality_history: list[QualityScore],
    final_score: QualityScore,
    audit_log: list[dict],
    generated_at: str,
) -> str:
    lines: list[str] = []

    def h(level: int, text: str) -> None:
        lines.append(f"\n{'#' * level} {text}\n")

    def row(*cells: str) -> str:
        return "| " + " | ".join(str(c) for c in cells) + " |"

    def sep(*widths: int) -> str:
        return "| " + " | ".join("-" * w for w in widths) + " |"

    # ── Title ──────────────────────────────────────────────────────────────
    lines.append(f"# Agentic Data Cleaner — Audit Report\n")
    lines.append(f"**File:** `{filename}`  ")
    lines.append(f"**Run ID:** `{run_id}`  ")
    lines.append(f"**Generated:** {generated_at}  \n")

    # ── Executive summary ──────────────────────────────────────────────────
    h(2, "Executive Summary")
    rows_before, cols_before = shape_before
    rows_after, cols_after = shape_after
    iters = len(quality_history)

    lines.append(f"The pipeline processed **{rows_before:,} rows × {cols_before} columns** "
                 f"and completed in **{iters} iteration(s)**.\n")
    lines.append(f"The final data quality score is **{final_score['overall'] * 100:.1f}%**.\n")
    lines.append("")
    lines.append(row("Metric", "Before", "After"))
    lines.append(sep(30, 12, 12))
    lines.append(row("Rows", f"{rows_before:,}", f"{rows_after:,}"))
    lines.append(row("Columns", cols_before, cols_after))
    lines.append(row("Missing score", "—", _score_bar(final_score["missing_score"])))
    lines.append(row("Outlier score", "—", _score_bar(final_score["outlier_score"])))
    lines.append(row("Skewness score", "—", _score_bar(final_score["skewness_score"])))
    lines.append(row("**Overall quality**", "—", f"**{_score_bar(final_score['overall'])}**"))
    lines.append("")

    # ── Quality history ────────────────────────────────────────────────────
    h(2, "Quality Score History")
    lines.append(row("Iteration", "Overall", "Missing", "Outlier", "Skewness"))
    lines.append(sep(10, 24, 24, 24, 24))
    for qs in quality_history:
        lines.append(row(
            qs.get("iteration", "?"),
            _score_bar(qs["overall"], 12),
            _score_bar(qs["missing_score"], 12),
            _score_bar(qs["outlier_score"], 12),
            _score_bar(qs["skewness_score"], 12),
        ))
    lines.append("")

    # ── Column profiles ────────────────────────────────────────────────────
    h(2, "Initial Column Profiles")
    lines.append(row("Column", "Type", "Missing", "Outlier ratio", "Skewness", "Cardinality"))
    lines.append(sep(20, 12, 10, 14, 12, 12))
    for p in initial_profile:
        skew = p.get("skewness")
        skew_str = f"{skew:.3f}" if skew is not None else "n/a"
        lines.append(row(
            f"`{p['column']}`",
            p.get("dtype", "?"),
            _pct(p.get("missing_pct")),
            _pct(p.get("outlier_ratio")),
            skew_str,
            p.get("cardinality_hint", "?"),
        ))
    lines.append("")

    # ── Cleaning plan with rationale ───────────────────────────────────────
    h(2, "Cleaning Plan & Rationale")
    lines.append(
        "> Each action below was chosen by the Decision Agent (LLM or rules engine) "
        "based on the column profile statistics.\n"
    )
    plan_by_col = {s["column"]: s for s in cleaning_plan}
    for p in initial_profile:
        col = p["column"]
        step = plan_by_col.get(col)
        if step is None:
            continue
        action = step.get("action", "unknown")
        rationale = step.get("rationale", "No rationale recorded.")
        params = step.get("params") or {}
        params_str = f" `{json.dumps(params)}`" if params else ""
        lines.append(f"### `{col}`\n")
        lines.append(f"**Action:** `{action}`{params_str}  ")
        lines.append(f"**Rationale:** {rationale}\n")

    # ── Execution log ──────────────────────────────────────────────────────
    h(2, "Execution Log")
    exec_entries = [e for e in audit_log if e.get("step") == "execution"]
    if not exec_entries:
        lines.append("_No execution entries found in audit log._\n")
    else:
        for exec_entry in exec_entries:
            iteration_n = exec_entry.get("iteration", "?")
            lines.append(f"**Iteration {iteration_n}** — "
                         f"shape {exec_entry.get('shape_before')} → "
                         f"{exec_entry.get('shape_after')} | "
                         f"ok={exec_entry.get('steps_ok', 0)} "
                         f"skipped={exec_entry.get('steps_skipped', 0)} "
                         f"error={exec_entry.get('steps_error', 0)}\n")

            step_log: list[dict] = exec_entry.get("step_log", [])
            if step_log:
                lines.append(row("Column", "Action", "Status", "Nulls before→after"))
                lines.append(sep(20, 18, 10, 18))
                for sl in step_log:
                    nb = sl.get("nulls_before", "")
                    na = sl.get("nulls_after", "")
                    null_change = f"{nb} → {na}" if nb != "" else "—"
                    lines.append(row(
                        f"`{sl['column']}`",
                        f"`{sl['action']}`",
                        sl.get("status", "?"),
                        null_change,
                    ))
                lines.append("")

    # ── Full audit log (collapsed) ─────────────────────────────────────────
    h(2, "Full Audit Log")
    lines.append("<details><summary>Expand raw audit log</summary>\n")
    lines.append("```json")
    # Exclude step_log arrays from the raw dump to keep it readable
    slim_log = []
    for entry in audit_log:
        slim = {k: v for k, v in entry.items() if k != "step_log"}
        slim_log.append(slim)
    lines.append(json.dumps(slim_log, indent=2, default=str))
    lines.append("```\n</details>\n")

    # ── Footer ─────────────────────────────────────────────────────────────
    lines.append("---")
    lines.append("*Generated by Agentic Data Cleaner — "
                 "github.com/your-username/agentic-data-cleaner*")

    return "\n".join(lines)


def _build_explainability_json(
    run_id: str,
    filename: str,
    shape_before: list[int],
    shape_after: list[int],
    initial_profile: list[ColumnProfile],
    cleaning_plan: list[CleaningStep],
    quality_history: list[QualityScore],
    final_score: QualityScore,
    audit_log: list[dict],
    iterations_completed: int,
) -> dict:
    """
    Build the structured payload the React frontend consumes.

    Returned dict matches the shape documented at the top of this file.
    """
    # Extract per-iteration execution step logs for the history tab
    iteration_history = []
    for entry in audit_log:
        if entry.get("step") == "execution":
            iteration_history.append({
                "iteration": entry.get("iteration"),
                "shape_before": entry.get("shape_before"),
                "shape_after": entry.get("shape_after"),
                "steps_ok": entry.get("steps_ok", 0),
                "steps_skipped": entry.get("steps_skipped", 0),
                "steps_error": entry.get("steps_error", 0),
                "step_log": entry.get("step_log", []),
            })

    return {
        "run_id": run_id,
        "filename": filename,
        "shape_before": shape_before,
        "shape_after": shape_after,
        "iterations_completed": iterations_completed,
        "final_score": dict(final_score),
        "quality_history": [dict(qs) for qs in quality_history],
        "cleaning_plan": [dict(s) for s in cleaning_plan],
        "column_profiles": [dict(p) for p in initial_profile],
        "iteration_history": iteration_history,
    }


# ---------------------------------------------------------------------------
# Shape extraction helper (reads ingestion audit entry)
# ---------------------------------------------------------------------------

def _extract_shapes(audit_log: list[dict]) -> tuple[list[int], list[int]]:
    """
    Pull shape_before (from ingestion) and shape_after (from last execution)
    out of the audit log.  Falls back to [0, 0] if entries are missing.
    """
    shape_before = [0, 0]
    shape_after = [0, 0]

    for entry in audit_log:
        if entry.get("step") == "ingestion" and entry.get("status") == "ok":
            shape_before = entry.get("shape_before", [0, 0])
        if entry.get("step") == "execution" and entry.get("status") == "ok":
            shape_after = entry.get("shape_after", [0, 0])

    return shape_before, shape_after


# ---------------------------------------------------------------------------
# Public node function
# ---------------------------------------------------------------------------

def run_output(state: PipelineState) -> dict[str, Any]:
    """
    LangGraph node: output.

    Reads  : entire state
    Writes : state["output_paths"], state["explainability"], state["audit_log"]
    """
    audit_log: list[dict] = list(state.get("audit_log", []))
    entry: dict[str, Any] = {
        "step": "output",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        df: pd.DataFrame = state["current_dataset"]
        filename: str = state.get("filename", "dataset.csv")
        cleaning_plan: list[CleaningStep] = state.get("cleaning_plan") or []
        initial_profile: list[ColumnProfile] = state.get("profile") or []
        quality_history: list[QualityScore] = state.get("quality_history") or []
        final_score: QualityScore = state.get("quality_score") or QualityScore(
            overall=0.0, missing_score=0.0, outlier_score=0.0,
            skewness_score=0.0, iteration=0, details={},
        )
        iteration: int = state.get("iteration", 0)

        if df is None or df.empty:
            raise ValueError("current_dataset is empty — nothing to output.")

        # --- Run ID & output directory --------------------------------------
        run_id = _make_run_id(filename)
        out_dir = _OUTPUT_ROOT / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- Shapes ---------------------------------------------------------
        shape_before, shape_after = _extract_shapes(audit_log)
        # Fallback: if execution log is missing, use df.shape as shape_after
        if shape_after == [0, 0]:
            shape_after = list(df.shape)

        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # --- 1. Cleaned CSV -------------------------------------------------
        csv_path = out_dir / f"cleaned_{Path(filename).stem}.csv"
        csv_bytes = _build_csv_bytes(df)
        csv_path.write_bytes(csv_bytes)
        logger.info("Wrote cleaned CSV → %s (%d bytes)", csv_path, len(csv_bytes))

        # --- 2. Markdown audit report ---------------------------------------
        md_path = out_dir / "audit_report.md"
        md_content = _build_markdown_report(
            filename=filename,
            run_id=run_id,
            shape_before=shape_before,
            shape_after=shape_after,
            initial_profile=initial_profile,
            cleaning_plan=cleaning_plan,
            quality_history=quality_history,
            final_score=final_score,
            audit_log=audit_log,
            generated_at=generated_at,
        )
        md_path.write_text(md_content, encoding="utf-8")
        logger.info("Wrote audit report → %s", md_path)

        # --- 3. Explainability JSON -----------------------------------------
        json_path = out_dir / "explainability.json"
        explain_payload = _build_explainability_json(
            run_id=run_id,
            filename=filename,
            shape_before=shape_before,
            shape_after=shape_after,
            initial_profile=initial_profile,
            cleaning_plan=cleaning_plan,
            quality_history=quality_history,
            final_score=final_score,
            audit_log=audit_log,
            iterations_completed=iteration,
        )
        json_path.write_text(
            json.dumps(explain_payload, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("Wrote explainability JSON → %s", json_path)

        output_paths = {
            "csv":   str(csv_path),
            "report": str(md_path),
            "explainability": str(json_path),
            "run_id": run_id,
            "out_dir": str(out_dir),
        }

        entry.update({
            "status": "ok",
            "run_id": run_id,
            "csv_bytes": len(csv_bytes),
            "output_paths": output_paths,
        })

    except Exception as exc:  # noqa: BLE001
        entry.update({"status": "error", "error": str(exc)})
        logger.exception("Output agent failed.")
        audit_log.append(entry)
        raise

    audit_log.append(entry)
    return {
        "output_paths": output_paths,
        "explainability": explain_payload,
        "audit_log": audit_log,
    }