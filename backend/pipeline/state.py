"""
backend/pipeline/state.py

PipelineState and all associated TypedDicts.
"""

from __future__ import annotations

from typing import Any, List, Optional

import pandas as pd
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Sub-schemas
# ---------------------------------------------------------------------------

class NumericSummary(TypedDict, total=False):
    mean: Optional[float]
    std: Optional[float]
    min: Optional[float]
    max: Optional[float]


class ColumnProfile(TypedDict, total=False):
    column: str
    dtype: str
    missing_pct: Optional[float]       # 0.0 – 1.0
    unique_count: Optional[int]
    cardinality_hint: str              # binary | low | medium | high | identifier
    skewness: Optional[float]          # numeric only; None for categorical
    outlier_ratio: Optional[float]     # numeric only; IQR method
    sample_values: List[Any]           # first 3 non-null values
    numeric_summary: Optional[NumericSummary]
    profiling_error: str               # only present if profiling failed for this col


class CleaningStep(TypedDict, total=False):
    column: str
    action: str            # e.g. "impute_median", "drop", "encode_onehot", "log_transform"
    rationale: str         # LLM-generated explanation
    params: dict           # action-specific parameters


class QualityScore(TypedDict, total=False):
    overall: float                     # 0.0 – 1.0 composite score
    missing_score: float               # 1 - avg missing_pct
    outlier_score: float               # 1 - avg outlier_ratio
    skewness_score: float              # penalty for highly skewed columns
    iteration: int                     # which iteration produced this score
    details: dict                      # per-column breakdown


# ---------------------------------------------------------------------------
# Top-level pipeline state
# ---------------------------------------------------------------------------

class PipelineState(TypedDict, total=False):
    # Data
    raw_dataset: Optional[pd.DataFrame]
    current_dataset: Optional[pd.DataFrame]
    filename: str

    # Agent outputs
    profile: Optional[List[ColumnProfile]]
    cleaning_plan: Optional[List[CleaningStep]]
    quality_score: Optional[QualityScore]

    # Loop control
    iteration: int
    max_iterations: int          # default 3
    quality_threshold: float     # default 0.90

    # History & logging
    audit_log: List[dict]
    quality_history: List[QualityScore]

    # Auth / mode
    api_key: str
    mode: str                    # "llm" | "rules"

    # Output agent artefacts (written by run_output)
    output_paths: Optional[dict]   # {csv, report, explainability, run_id, out_dir}
    explainability: Optional[dict] # full payload for the frontend