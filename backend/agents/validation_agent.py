"""
backend/agents/validation_agent.py

Validation Agent — Phase 4
Scores the current dataset after execution and decides whether the pipeline
loop should continue or terminate.

Quality score formula
---------------------
Three sub-scores are computed, each in [0, 1]:

  missing_score  = 1 - mean(missing_pct per numeric/categorical column)
  outlier_score  = 1 - mean(outlier_ratio per numeric column)
                   (columns with no outliers → 1.0)
  skewness_score = mean of per-column skewness scores, where
                   skew_score(col) = max(0, 1 - |skewness| / SKEW_PENALTY_CAP)
                   Skewness of 0 → 1.0; skewness of ±2 → 0.0 (cap at 2).

Overall score = weighted average:
  0.50 × missing_score + 0.30 × outlier_score + 0.20 × skewness_score

The weights reflect real-world priorities: missing data is the most
destructive quality issue for ML, followed by outliers, then skew.

The node also increments nothing — iteration counting lives in execution_agent.
The LangGraph conditional edge (should_continue in graph.py) reads
quality_score["overall"] and state["iteration"] to decide looping.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from pipeline.state import PipelineState, QualityScore

logger = logging.getLogger(__name__)

# Weight vector for the three sub-scores
_W_MISSING  = 0.50
_W_OUTLIER  = 0.30
_W_SKEWNESS = 0.20

# Skewness above this magnitude is treated as the worst possible score (0.0)
_SKEW_PENALTY_CAP = 2.0


# ---------------------------------------------------------------------------
# Sub-score helpers
# ---------------------------------------------------------------------------

def _col_missing_pct(series: pd.Series) -> float:
    return series.isna().sum() / max(len(series), 1)


def _col_outlier_ratio(series: pd.Series) -> float:
    """IQR method — mirrors profiling_agent exactly."""
    if not pd.api.types.is_numeric_dtype(series):
        return 0.0
    non_null = series.dropna()
    n = len(non_null)
    if n == 0:
        return 0.0
    q1 = float(non_null.quantile(0.25))
    q3 = float(non_null.quantile(0.75))
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return float(((non_null < lower) | (non_null > upper)).sum()) / n


def _col_skewness_score(series: pd.Series) -> float | None:
    """Returns a [0,1] score where 1 = perfectly symmetric, None = non-numeric."""
    if not pd.api.types.is_numeric_dtype(series):
        return None
    non_null = series.dropna()
    if len(non_null) < 3:
        return None
    skew = float(non_null.skew())
    if math.isnan(skew):
        return None
    # Linear penalty: |skew| == 0 → 1.0;  |skew| >= cap → 0.0
    return max(0.0, 1.0 - abs(skew) / _SKEW_PENALTY_CAP)


# ---------------------------------------------------------------------------
# Per-column detail record
# ---------------------------------------------------------------------------

def _column_detail(col: str, series: pd.Series) -> dict[str, Any]:
    missing = _col_missing_pct(series)
    outlier = _col_outlier_ratio(series)
    skew_score = _col_skewness_score(series)
    return {
        "column": col,
        "missing_pct": round(missing, 4),
        "outlier_ratio": round(outlier, 4),
        "skewness_score": round(skew_score, 4) if skew_score is not None else None,
    }


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def compute_quality_score(df: pd.DataFrame, iteration: int) -> QualityScore:
    """
    Compute a QualityScore for the given DataFrame.

    Returns a fully-populated QualityScore TypedDict.
    """
    if df.empty:
        return QualityScore(
            overall=0.0,
            missing_score=0.0,
            outlier_score=0.0,
            skewness_score=0.0,
            iteration=iteration,
            details={},
        )

    details: dict[str, Any] = {}
    missing_vals: list[float] = []
    outlier_vals: list[float] = []
    skewness_vals: list[float] = []

    for col in df.columns:
        detail = _column_detail(col, df[col])
        details[col] = detail

        missing_vals.append(detail["missing_pct"])
        outlier_vals.append(detail["outlier_ratio"])
        if detail["skewness_score"] is not None:
            skewness_vals.append(detail["skewness_score"])

    # Sub-scores
    missing_score = 1.0 - (sum(missing_vals) / max(len(missing_vals), 1))
    outlier_score = 1.0 - (sum(outlier_vals) / max(len(outlier_vals), 1))
    skewness_score = (
        sum(skewness_vals) / len(skewness_vals) if skewness_vals else 1.0
    )

    # Clamp to [0, 1] — floating-point arithmetic can produce tiny overflows
    missing_score  = max(0.0, min(1.0, missing_score))
    outlier_score  = max(0.0, min(1.0, outlier_score))
    skewness_score = max(0.0, min(1.0, skewness_score))

    overall = (
        _W_MISSING  * missing_score
        + _W_OUTLIER  * outlier_score
        + _W_SKEWNESS * skewness_score
    )
    overall = max(0.0, min(1.0, overall))

    return QualityScore(
        overall=round(overall, 4),
        missing_score=round(missing_score, 4),
        outlier_score=round(outlier_score, 4),
        skewness_score=round(skewness_score, 4),
        iteration=iteration,
        details=details,
    )


# ---------------------------------------------------------------------------
# Public node function
# ---------------------------------------------------------------------------

def run_validation(state: PipelineState) -> dict[str, Any]:
    """
    LangGraph node: validation.

    Reads  : state["current_dataset"], state["iteration"]
    Writes : state["quality_score"], state["quality_history"], state["audit_log"]

    The node does NOT decide whether to loop — that is the job of the
    conditional edge function should_continue() in graph.py.
    """
    audit_log: list[dict] = list(state.get("audit_log", []))
    quality_history: list[QualityScore] = list(state.get("quality_history", []))
    iteration: int = state.get("iteration", 0)

    entry: dict[str, Any] = {
        "step": "validation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "iteration": iteration,
    }

    try:
        df: pd.DataFrame = state["current_dataset"]

        if df is None or df.empty:
            raise ValueError("current_dataset is empty — cannot validate.")

        score = compute_quality_score(df, iteration)

        threshold: float = state.get("quality_threshold", 0.90)
        max_iter: int = state.get("max_iterations", 3)
        passed = score["overall"] >= threshold
        exhausted = iteration >= max_iter

        entry.update({
            "status": "ok",
            "overall": score["overall"],
            "missing_score": score["missing_score"],
            "outlier_score": score["outlier_score"],
            "skewness_score": score["skewness_score"],
            "threshold": threshold,
            "passed": passed,
            "loop_exhausted": exhausted,
            "verdict": "done" if (passed or exhausted) else "retry",
        })
        logger.info(
            "Validation (iter %d) — overall=%.4f | missing=%.4f outlier=%.4f "
            "skew=%.4f | verdict=%s",
            iteration,
            score["overall"],
            score["missing_score"],
            score["outlier_score"],
            score["skewness_score"],
            entry["verdict"],
        )

    except Exception as exc:  # noqa: BLE001
        entry.update({"status": "error", "error": str(exc)})
        logger.exception("Validation agent failed.")
        audit_log.append(entry)
        raise

    quality_history.append(score)
    audit_log.append(entry)

    return {
        "quality_score": score,
        "quality_history": quality_history,
        "audit_log": audit_log,
    }