"""
backend/agents/execution_agent.py

Execution Agent — Phase 4
Applies the cleaning_plan produced by the Decision Agent to current_dataset.

Design principles
-----------------
- One handler function per action, registered in HANDLERS dict.
- Each handler receives (df, col, params) and returns a modified DataFrame.
- Handlers are pure: they never mutate the input, always return a copy.
- Unknown / skipped actions are logged and left as-is (never raise).
- A per-step record is appended to the execution audit so the frontend can
  show exactly what happened column by column.
- Sklearn transformers are fit on the non-null slice and applied in-place so
  that NaN rows (imputed earlier) are not excluded from scaling.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from pipeline.state import CleaningStep, PipelineState

logger = logging.getLogger(__name__)

# Type alias for a handler function
Handler = Callable[[pd.DataFrame, str, dict], pd.DataFrame]


# ---------------------------------------------------------------------------
# Individual action handlers
# Each handler signature:  (df, column, params) -> df
# ---------------------------------------------------------------------------

def _impute_mean(df: pd.DataFrame, col: str, params: dict) -> pd.DataFrame:
    fill = df[col].mean()
    df[col] = df[col].fillna(fill)
    return df


def _impute_median(df: pd.DataFrame, col: str, params: dict) -> pd.DataFrame:
    fill = df[col].median()
    df[col] = df[col].fillna(fill)
    return df


def _impute_mode(df: pd.DataFrame, col: str, params: dict) -> pd.DataFrame:
    mode_series = df[col].mode()
    if mode_series.empty:
        return df                       # nothing to fill with
    df[col] = df[col].fillna(mode_series.iloc[0])
    return df


def _impute_constant(df: pd.DataFrame, col: str, params: dict) -> pd.DataFrame:
    value = params.get("value", 0)
    df[col] = df[col].fillna(value)
    return df


def _drop_column(df: pd.DataFrame, col: str, params: dict) -> pd.DataFrame:
    return df.drop(columns=[col])


def _clip_iqr(df: pd.DataFrame, col: str, params: dict) -> pd.DataFrame:
    factor = float(params.get("factor", 1.5))
    non_null = df[col].dropna()
    if non_null.empty:
        return df
    q1 = non_null.quantile(0.25)
    q3 = non_null.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def _winsorise(df: pd.DataFrame, col: str, params: dict) -> pd.DataFrame:
    lower_pct = float(params.get("lower", 0.01))
    upper_pct = float(params.get("upper", 0.99))
    lower = df[col].quantile(lower_pct)
    upper = df[col].quantile(upper_pct)
    df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def _encode_onehot(df: pd.DataFrame, col: str, params: dict) -> pd.DataFrame:
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=float)
    df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
    return df


def _encode_ordinal(df: pd.DataFrame, col: str, params: dict) -> pd.DataFrame:
    order: list = params.get("order", [])
    if not order:
        # Fallback: alphabetical order
        order = sorted(df[col].dropna().unique().tolist())
    mapping = {v: i for i, v in enumerate(order)}
    df[col] = df[col].map(mapping)
    return df


def _encode_binary(df: pd.DataFrame, col: str, params: dict) -> pd.DataFrame:
    """
    Maps a binary string column to 0/1.
    Works for common patterns: yes/no, true/false, 1/0 stored as strings.
    Falls back to ordinal encoding on unknown patterns.
    """
    unique_vals = set(df[col].dropna().str.lower().unique())
    positive = {"yes", "true", "1", "y", "t"}
    if unique_vals <= (positive | {"no", "false", "0", "n", "f"}):
        df[col] = df[col].str.lower().map(
            lambda x: 1 if x in positive else (0 if pd.notna(x) else np.nan)
        )
    else:
        # Generic fallback
        df = _encode_ordinal(df, col, params)
    return df


def _scale_standard(df: pd.DataFrame, col: str, params: dict) -> pd.DataFrame:
    scaler = StandardScaler()
    mask = df[col].notna()
    df.loc[mask, col] = scaler.fit_transform(df.loc[mask, col].values.reshape(-1, 1)).flatten()
    return df


def _scale_minmax(df: pd.DataFrame, col: str, params: dict) -> pd.DataFrame:
    scaler = MinMaxScaler()
    mask = df[col].notna()
    df.loc[mask, col] = scaler.fit_transform(df.loc[mask, col].values.reshape(-1, 1)).flatten()
    return df


def _scale_robust(df: pd.DataFrame, col: str, params: dict) -> pd.DataFrame:
    scaler = RobustScaler()
    mask = df[col].notna()
    df.loc[mask, col] = scaler.fit_transform(df.loc[mask, col].values.reshape(-1, 1)).flatten()
    return df


def _log_transform(df: pd.DataFrame, col: str, params: dict) -> pd.DataFrame:
    """log1p transform.  Clips negatives to 0 first as a safety net."""
    df[col] = np.log1p(df[col].clip(lower=0))
    return df


def _sqrt_transform(df: pd.DataFrame, col: str, params: dict) -> pd.DataFrame:
    """sqrt transform.  Clips negatives to 0 first."""
    df[col] = np.sqrt(df[col].clip(lower=0))
    return df


def _keep(df: pd.DataFrame, col: str, params: dict) -> pd.DataFrame:
    return df  # no-op


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

HANDLERS: dict[str, Handler] = {
    "impute_mean":      _impute_mean,
    "impute_median":    _impute_median,
    "impute_mode":      _impute_mode,
    "impute_constant":  _impute_constant,
    "drop_column":      _drop_column,
    "clip_iqr":         _clip_iqr,
    "winsorise":        _winsorise,
    "encode_onehot":    _encode_onehot,
    "encode_ordinal":   _encode_ordinal,
    "encode_binary":    _encode_binary,
    "scale_standard":   _scale_standard,
    "scale_minmax":     _scale_minmax,
    "scale_robust":     _scale_robust,
    "log_transform":    _log_transform,
    "sqrt_transform":   _sqrt_transform,
    "keep":             _keep,
}


# ---------------------------------------------------------------------------
# Core execution loop
# ---------------------------------------------------------------------------

def apply_plan(
    df: pd.DataFrame,
    plan: list[CleaningStep],
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Apply every step in *plan* to *df* in order.

    Returns
    -------
    df          : transformed DataFrame
    step_log    : list of per-step audit records
    """
    df = df.copy()
    step_log: list[dict] = []

    # Index plan by column for O(1) lookup; duplicates are applied in order
    for step in plan:
        col: str = step["column"]
        action: str = step["action"]
        params: dict = step.get("params") or {}

        record: dict[str, Any] = {
            "column": col,
            "action": action,
            "params": params,
        }

        # --- column may have been dropped by an earlier step ----------------
        if col not in df.columns and action != "drop_column":
            record["status"] = "skipped"
            record["reason"] = "column no longer exists (dropped earlier)"
            step_log.append(record)
            continue

        handler = HANDLERS.get(action)
        if handler is None:
            record["status"] = "skipped"
            record["reason"] = f"unknown action '{action}'"
            logger.warning("Unknown action '%s' for column '%s' — skipped.", action, col)
            step_log.append(record)
            continue

        try:
            before_null = int(df[col].isna().sum()) if col in df.columns else 0
            before_shape = list(df.shape)

            df = handler(df, col, params)

            after_shape = list(df.shape)
            after_null = int(df[col].isna().sum()) if col in df.columns else 0

            record["status"] = "ok"
            record["nulls_before"] = before_null
            record["nulls_after"] = after_null
            record["shape_before"] = before_shape
            record["shape_after"] = after_shape

        except Exception as exc:  # noqa: BLE001
            record["status"] = "error"
            record["error"] = str(exc)
            logger.error(
                "Error applying '%s' to column '%s': %s", action, col, exc
            )

        step_log.append(record)

    return df, step_log


# ---------------------------------------------------------------------------
# Public node function
# ---------------------------------------------------------------------------

def run_execution(state: PipelineState) -> dict[str, Any]:
    """
    LangGraph node: execution.

    Reads  : state["current_dataset"], state["cleaning_plan"], state["iteration"]
    Writes : state["current_dataset"], state["iteration"], state["audit_log"]
    """
    audit_log: list[dict] = list(state.get("audit_log", []))
    iteration: int = state.get("iteration", 0)
    entry: dict[str, Any] = {
        "step": "execution",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "iteration": iteration,
    }

    try:
        df: pd.DataFrame = state["current_dataset"]
        plan: list[CleaningStep] = state.get("cleaning_plan") or []

        if df is None or df.empty:
            raise ValueError("current_dataset is empty — nothing to execute.")
        if not plan:
            raise ValueError("cleaning_plan is empty — decision agent must run first.")

        shape_before = list(df.shape)
        df, step_log = apply_plan(df, plan)
        shape_after = list(df.shape)

        n_ok = sum(1 for s in step_log if s["status"] == "ok")
        n_skipped = sum(1 for s in step_log if s["status"] == "skipped")
        n_error = sum(1 for s in step_log if s["status"] == "error")

        entry.update({
            "status": "ok",
            "shape_before": shape_before,
            "shape_after": shape_after,
            "steps_ok": n_ok,
            "steps_skipped": n_skipped,
            "steps_error": n_error,
            "step_log": step_log,
        })
        logger.info(
            "Execution OK (iter %d) — shape %s → %s | ok=%d skipped=%d error=%d",
            iteration, shape_before, shape_after, n_ok, n_skipped, n_error,
        )

    except Exception as exc:  # noqa: BLE001
        entry.update({"status": "error", "error": str(exc)})
        logger.exception("Execution agent failed.")
        audit_log.append(entry)
        raise

    audit_log.append(entry)
    return {
        "current_dataset": df,
        "iteration": iteration + 1,
        "audit_log": audit_log,
    }