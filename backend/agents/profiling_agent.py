"""
backend/agents/profiling_agent.py

Profiling Agent — Phase 2
Iterates over every column in state["current_dataset"] and produces a rich
ColumnProfile for each one.  Results are written to state["profile"].
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from pipeline.state import ColumnProfile, PipelineState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-column profilers
# ---------------------------------------------------------------------------

def _missing_pct(series: pd.Series) -> float:
    """Fraction of null values, rounded to 4 dp."""
    return round(series.isna().sum() / max(len(series), 1), 4)


def _unique_count(series: pd.Series) -> int:
    return int(series.nunique(dropna=True))


def _skewness(series: pd.Series) -> float | None:
    """
    Returns Fisher-Pearson skewness for numeric columns.
    Returns None for non-numeric or when fewer than 3 non-null values exist
    (pandas returns NaN in that case anyway).
    """
    if not pd.api.types.is_numeric_dtype(series):
        return None
    non_null = series.dropna()
    if len(non_null) < 3:
        return None
    skew = float(non_null.skew())
    return None if math.isnan(skew) else round(skew, 4)


def _outlier_ratio(series: pd.Series) -> float | None:
    """
    IQR-based outlier ratio for numeric columns.
    A value is an outlier if it lies outside [Q1 - 1.5·IQR, Q3 + 1.5·IQR].
    Returns None for non-numeric columns.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return None

    non_null = series.dropna()
    n = len(non_null)
    if n == 0:
        return None

    q1 = float(non_null.quantile(0.25))
    q3 = float(non_null.quantile(0.75))
    iqr = q3 - q1

    if iqr == 0:
        # All values are identical — no outliers by IQR definition
        return 0.0

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    n_outliers = int(((non_null < lower) | (non_null > upper)).sum())
    return round(n_outliers / n, 4)


def _sample_values(series: pd.Series, n: int = 3) -> list[Any]:
    """First *n* non-null values, JSON-serialisable."""
    samples = series.dropna().head(n).tolist()
    safe = []
    for v in samples:
        if isinstance(v, (np.integer,)):
            safe.append(int(v))
        elif isinstance(v, (np.floating,)):
            safe.append(None if math.isnan(float(v)) else float(v))
        elif isinstance(v, (pd.Timestamp,)):
            safe.append(v.isoformat())
        else:
            safe.append(str(v))
    return safe


def _cardinality_hint(series: pd.Series, unique_count: int) -> str:
    """
    Rough cardinality label — useful context for the Decision Agent.
      binary      : 2 unique values
      low         : ≤ 10 unique values
      medium      : 11–50 unique values
      high        : > 50 unique values
      identifier  : unique_count ≈ row count (likely an ID column)
    """
    n_rows = len(series.dropna())
    if unique_count <= 2:
        return "binary"
    if unique_count / max(n_rows, 1) > 0.95:
        return "identifier"
    if unique_count <= 10:
        return "low"
    if unique_count <= 50:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Summary statistics (numeric columns)
# ---------------------------------------------------------------------------

def _numeric_summary(series: pd.Series) -> dict[str, float | None] | None:
    """Mean, std, min, max for numeric columns. Returns None otherwise."""
    if not pd.api.types.is_numeric_dtype(series):
        return None
    non_null = series.dropna()
    if len(non_null) == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": round(float(non_null.mean()), 4),
        "std": round(float(non_null.std()), 4),
        "min": round(float(non_null.min()), 4),
        "max": round(float(non_null.max()), 4),
    }


# ---------------------------------------------------------------------------
# Main profile builder
# ---------------------------------------------------------------------------

def _profile_column(col_name: str, series: pd.Series) -> ColumnProfile:
    """Build a ColumnProfile dict for a single series."""
    unique_count = _unique_count(series)

    profile: ColumnProfile = {
        "column": col_name,
        "dtype": str(series.dtype),
        "missing_pct": _missing_pct(series),
        "unique_count": unique_count,
        "cardinality_hint": _cardinality_hint(series, unique_count),
        "skewness": _skewness(series),
        "outlier_ratio": _outlier_ratio(series),
        "sample_values": _sample_values(series),
        "numeric_summary": _numeric_summary(series),
    }
    return profile


# ---------------------------------------------------------------------------
# Public node function
# ---------------------------------------------------------------------------

def run_profiling(state: PipelineState) -> dict[str, Any]:
    """
    LangGraph node: profiling.

    Reads  : state["current_dataset"]
    Writes : state["profile"], state["audit_log"]
    """
    audit_log: list[dict] = list(state.get("audit_log", []))
    entry: dict[str, Any] = {
        "step": "profiling",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        df: pd.DataFrame = state["current_dataset"]

        if df is None or df.empty:
            raise ValueError("current_dataset is empty — cannot profile.")

        profiles: list[ColumnProfile] = []
        for col in df.columns:
            try:
                col_profile = _profile_column(col, df[col])
                profiles.append(col_profile)
            except Exception as col_exc:  # noqa: BLE001
                logger.warning("Could not profile column '%s': %s", col, col_exc)
                # Include a minimal stub so downstream agents know the column exists
                profiles.append({
                    "column": col,
                    "dtype": str(df[col].dtype),
                    "missing_pct": None,
                    "unique_count": None,
                    "cardinality_hint": "unknown",
                    "skewness": None,
                    "outlier_ratio": None,
                    "sample_values": [],
                    "numeric_summary": None,
                    "profiling_error": str(col_exc),
                })

        # --- dataset-level summary for the audit log -----------------------
        n_rows, n_cols = df.shape
        numeric_cols = [p["column"] for p in profiles if p.get("skewness") is not None]
        high_missing = [p["column"] for p in profiles if (p.get("missing_pct") or 0) > 0.2]
        high_skew = [
            p["column"] for p in profiles
            if p.get("skewness") is not None and abs(p["skewness"]) > 1.0
        ]
        high_outliers = [
            p["column"] for p in profiles
            if p.get("outlier_ratio") is not None and p["outlier_ratio"] > 0.05
        ]

        entry.update({
            "status": "ok",
            "n_rows": n_rows,
            "n_cols": n_cols,
            "numeric_cols": numeric_cols,
            "high_missing_cols": high_missing,
            "high_skew_cols": high_skew,
            "high_outlier_cols": high_outliers,
            "profiles_generated": len(profiles),
        })

        logger.info(
            "Profiling OK — %d cols | %d numeric | %d high-missing | "
            "%d high-skew | %d high-outlier",
            n_cols, len(numeric_cols), len(high_missing),
            len(high_skew), len(high_outliers),
        )

    except Exception as exc:  # noqa: BLE001
        entry.update({"status": "error", "error": str(exc)})
        logger.exception("Profiling failed.")
        audit_log.append(entry)
        raise

    audit_log.append(entry)
    return {
        "profile": profiles,
        "audit_log": audit_log,
    }