"""
backend/agents/ingestion.py

Ingestion Agent — Phase 2
Receives a raw DataFrame from state, coerces dtypes, validates structure,
and writes the cleaned frame to state["current_dataset"].
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from pipeline.state import PipelineState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_numeric_coercion(series: pd.Series) -> pd.Series:
    """
    If a column is object-typed but every non-null value parses as a number,
    coerce it to a numeric dtype.  Returns the original series unchanged if
    coercion would introduce new NaNs (i.e. some values genuinely aren't numeric).
    """
    if series.dtype != object:
        return series

    coerced = pd.to_numeric(series, errors="coerce")
    original_nulls = series.isna().sum()
    new_nulls = coerced.isna().sum()

    if new_nulls == original_nulls:          # no extra NaNs introduced
        return coerced
    return series                            # leave as-is — mixed/text column


def _try_datetime_coercion(series: pd.Series) -> pd.Series:
    """
    Attempt to parse object columns that look like dates.  Only coerces when
    every non-null value parses cleanly so we never silently destroy data.
    """
    if series.dtype != object:
        return series

    try:
        coerced = pd.to_datetime(series, errors="coerce")
        original_nulls = series.isna().sum()
        new_nulls = coerced.isna().sum()
        if new_nulls == original_nulls:
            return coerced
    except Exception:
        pass
    return series


def _coerce_dtypes(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Walk every column and attempt safe dtype coercions.
    Returns the (possibly modified) DataFrame and a list of change-records.
    """
    changes: list[dict] = []
    df = df.copy()

    for col in df.columns:
        original_dtype = str(df[col].dtype)

        # 1. Numeric coercion
        df[col] = _try_numeric_coercion(df[col])
        if str(df[col].dtype) != original_dtype:
            changes.append({"column": col, "from": original_dtype, "to": str(df[col].dtype), "method": "numeric"})
            continue

        # 2. Datetime coercion (only for object columns not already numeric)
        df[col] = _try_datetime_coercion(df[col])
        if str(df[col].dtype) != original_dtype:
            changes.append({"column": col, "from": original_dtype, "to": str(df[col].dtype), "method": "datetime"})

    return df, changes


def _drop_fully_empty_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove rows where every value is null. Returns (df, rows_dropped)."""
    mask = df.isna().all(axis=1)
    dropped = int(mask.sum())
    return df[~mask].reset_index(drop=True), dropped


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class IngestionError(ValueError):
    """Raised when the dataset fails a hard validation check."""


def _validate(df: pd.DataFrame) -> None:
    """
    Hard-fail checks.  Raises IngestionError with a human-readable message
    so the caller can surface it to the user via the audit log.
    """
    if df is None:
        raise IngestionError("raw_dataset is None — nothing to ingest.")

    if df.empty:
        raise IngestionError("Dataset is empty (0 rows).")

    if df.shape[1] < 2:
        raise IngestionError(
            f"Dataset has only {df.shape[1]} column(s); at least 2 are required."
        )

    # Warn (but don't fail) if the dataset is very wide or very large
    if df.shape[0] > 1_000_000:
        logger.warning("Dataset has %d rows — processing may be slow.", df.shape[0])


# ---------------------------------------------------------------------------
# Public node function
# ---------------------------------------------------------------------------

def run_ingestion(state: PipelineState) -> dict[str, Any]:
    """
    LangGraph node: ingestion.

    Reads  : state["raw_dataset"]
    Writes : state["current_dataset"], state["audit_log"]
    """
    audit_log: list[dict] = list(state.get("audit_log", []))
    entry: dict[str, Any] = {
        "step": "ingestion",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        df: pd.DataFrame = state["raw_dataset"]

        # --- hard validation (pre-coercion) ---------------------------------
        _validate(df)
        shape_before = list(df.shape)

        # --- drop fully-empty rows ------------------------------------------
        df, rows_dropped = _drop_fully_empty_rows(df)

        # --- dtype coercion -------------------------------------------------
        df, dtype_changes = _coerce_dtypes(df)
        shape_after = list(df.shape)

        entry.update({
            "status": "ok",
            "shape_before": shape_before,
            "shape_after": shape_after,
            "fully_empty_rows_dropped": rows_dropped,
            "dtype_changes": dtype_changes,
            "columns": list(df.columns),
        })
        logger.info(
            "Ingestion OK — shape %s → %s, %d dtype change(s), %d empty row(s) dropped.",
            shape_before, shape_after, len(dtype_changes), rows_dropped,
        )

    except IngestionError as exc:
        entry.update({"status": "error", "error": str(exc)})
        logger.error("Ingestion failed: %s", exc)
        audit_log.append(entry)
        # Re-raise so LangGraph marks the run as failed
        raise

    except Exception as exc:  # noqa: BLE001
        entry.update({"status": "error", "error": f"Unexpected error: {exc}"})
        logger.exception("Unexpected ingestion error.")
        audit_log.append(entry)
        raise

    audit_log.append(entry)
    return {
        "current_dataset": df,
        "audit_log": audit_log,
    }