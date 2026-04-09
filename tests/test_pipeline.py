"""
tests/test_pipeline.py  — Phase 2 additions

Run with:  pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import math
import sys
import os

import numpy as np
import pandas as pd
import pytest

# Make sure backend/ is on sys.path when running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from agents.ingestion import run_ingestion, IngestionError
from agents.profiling_agent import run_profiling


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_state(df: pd.DataFrame, **overrides) -> dict:
    """Minimal PipelineState for testing."""
    base = {
        "raw_dataset": df,
        "current_dataset": None,
        "filename": "test.csv",
        "profile": None,
        "cleaning_plan": None,
        "quality_score": None,
        "iteration": 0,
        "max_iterations": 3,
        "quality_threshold": 0.90,
        "audit_log": [],
        "quality_history": [],
        "api_key": "test-key",
        "mode": "rules",
    }
    base.update(overrides)
    return base


@pytest.fixture
def clean_df():
    """A tidy, multi-column DataFrame with no issues."""
    return pd.DataFrame({
        "age": [25, 32, 47, 51, 62],
        "income": [50000.0, 72000.0, 91000.0, 115000.0, 88000.0],
        "name": ["Alice", "Bob", "Carol", "Dan", "Eve"],
        "active": [True, False, True, True, False],
    })


@pytest.fixture
def messy_df():
    """DataFrame with string-encoded numbers, nulls, and a fully-empty row."""
    return pd.DataFrame({
        "age": ["25", "32", None, "51", "62"],
        "income": ["50000", "72000", "91000", None, "88000"],
        "city": ["NYC", "LA", "Chicago", None, "NYC"],
        "score": [np.nan, np.nan, np.nan, np.nan, np.nan],  # all null col
    })


@pytest.fixture
def outlier_df():
    """DataFrame with clear IQR outliers."""
    values = [10, 11, 10, 12, 10, 11, 200, 9, 10, -150]  # 200 and -150 are outliers
    return pd.DataFrame({"value": values, "label": list("abcdefghij")})


# ---------------------------------------------------------------------------
# Ingestion tests
# ---------------------------------------------------------------------------

class TestIngestion:

    def test_happy_path_returns_current_dataset(self, clean_df):
        result = run_ingestion(_make_state(clean_df))
        assert "current_dataset" in result
        assert isinstance(result["current_dataset"], pd.DataFrame)
        assert result["current_dataset"].shape == clean_df.shape

    def test_audit_log_appended(self, clean_df):
        result = run_ingestion(_make_state(clean_df))
        assert len(result["audit_log"]) == 1
        assert result["audit_log"][0]["step"] == "ingestion"
        assert result["audit_log"][0]["status"] == "ok"

    def test_numeric_coercion(self, messy_df):
        result = run_ingestion(_make_state(messy_df))
        df_out = result["current_dataset"]
        # "age" and "income" were strings — should now be numeric
        assert pd.api.types.is_numeric_dtype(df_out["age"]), "age should be numeric"
        assert pd.api.types.is_numeric_dtype(df_out["income"]), "income should be numeric"
        # "city" is text — should remain object
        assert df_out["city"].dtype == object

    def test_dtype_changes_recorded(self, messy_df):
        result = run_ingestion(_make_state(messy_df))
        changes = result["audit_log"][0]["dtype_changes"]
        changed_cols = {c["column"] for c in changes}
        assert "age" in changed_cols
        assert "income" in changed_cols

    def test_fully_empty_rows_dropped(self):
        df = pd.DataFrame({
            "a": [1, np.nan, 3],
            "b": [4, np.nan, 6],
        })
        result = run_ingestion(_make_state(df))
        assert result["current_dataset"].shape[0] == 2
        assert result["audit_log"][0]["fully_empty_rows_dropped"] == 1

    def test_existing_audit_log_preserved(self, clean_df):
        prior_log = [{"step": "upload", "status": "ok"}]
        result = run_ingestion(_make_state(clean_df, audit_log=prior_log))
        assert len(result["audit_log"]) == 2
        assert result["audit_log"][0]["step"] == "upload"

    def test_raises_on_none_dataset(self):
        with pytest.raises(IngestionError):
            run_ingestion(_make_state(None))

    def test_raises_on_empty_dataframe(self):
        with pytest.raises(IngestionError):
            run_ingestion(_make_state(pd.DataFrame()))

    def test_raises_on_single_column(self):
        with pytest.raises(IngestionError):
            run_ingestion(_make_state(pd.DataFrame({"a": [1, 2, 3]})))

    def test_shape_before_after_recorded(self, messy_df):
        result = run_ingestion(_make_state(messy_df))
        log = result["audit_log"][0]
        assert "shape_before" in log
        assert "shape_after" in log


# ---------------------------------------------------------------------------
# Profiling tests
# ---------------------------------------------------------------------------

class TestProfiling:

    def _run(self, df: pd.DataFrame) -> dict:
        """Convenience: ingestion → profiling, returns profiling result."""
        ing = run_ingestion(_make_state(df))
        state_after_ing = {
            **_make_state(df),
            "current_dataset": ing["current_dataset"],
            "audit_log": ing["audit_log"],
        }
        return run_profiling(state_after_ing)

    def test_profile_length_matches_columns(self, clean_df):
        result = self._run(clean_df)
        assert len(result["profile"]) == clean_df.shape[1]

    def test_profile_column_names(self, clean_df):
        result = self._run(clean_df)
        profiled = {p["column"] for p in result["profile"]}
        assert profiled == set(clean_df.columns)

    def test_missing_pct_range(self, messy_df):
        result = self._run(messy_df)
        for p in result["profile"]:
            if p.get("missing_pct") is not None:
                assert 0.0 <= p["missing_pct"] <= 1.0

    def test_skewness_none_for_text_column(self, clean_df):
        result = self._run(clean_df)
        name_profile = next(p for p in result["profile"] if p["column"] == "name")
        assert name_profile["skewness"] is None

    def test_skewness_float_for_numeric_column(self, clean_df):
        result = self._run(clean_df)
        age_profile = next(p for p in result["profile"] if p["column"] == "age")
        assert age_profile["skewness"] is not None
        assert isinstance(age_profile["skewness"], float)

    def test_outlier_ratio_none_for_text_column(self, clean_df):
        result = self._run(clean_df)
        name_profile = next(p for p in result["profile"] if p["column"] == "name")
        assert name_profile["outlier_ratio"] is None

    def test_outlier_ratio_detects_outliers(self, outlier_df):
        result = self._run(outlier_df)
        value_profile = next(p for p in result["profile"] if p["column"] == "value")
        assert value_profile["outlier_ratio"] is not None
        assert value_profile["outlier_ratio"] > 0.0, "Should detect outliers in value column"

    def test_sample_values_non_null(self, clean_df):
        result = self._run(clean_df)
        for p in result["profile"]:
            for v in p["sample_values"]:
                # None means a null slipped through — should not happen
                assert v is not None or len(p["sample_values"]) == 0

    def test_sample_values_max_3(self, clean_df):
        result = self._run(clean_df)
        for p in result["profile"]:
            assert len(p["sample_values"]) <= 3

    def test_numeric_summary_present_for_numeric(self, clean_df):
        result = self._run(clean_df)
        age_profile = next(p for p in result["profile"] if p["column"] == "age")
        ns = age_profile["numeric_summary"]
        assert ns is not None
        assert "mean" in ns and "std" in ns and "min" in ns and "max" in ns

    def test_numeric_summary_none_for_text(self, clean_df):
        result = self._run(clean_df)
        name_profile = next(p for p in result["profile"] if p["column"] == "name")
        assert name_profile["numeric_summary"] is None

    def test_cardinality_hint_identifier(self):
        """Column where unique_count ≈ row count should be labelled 'identifier'."""
        df = pd.DataFrame({"id": range(100), "value": range(100)})
        result = self._run(df)
        id_profile = next(p for p in result["profile"] if p["column"] == "id")
        assert id_profile["cardinality_hint"] == "identifier"

    def test_cardinality_hint_binary(self):
        df = pd.DataFrame({"flag": [0, 1, 0, 1, 1], "x": [1, 2, 3, 4, 5]})
        result = self._run(df)
        flag_profile = next(p for p in result["profile"] if p["column"] == "flag")
        assert flag_profile["cardinality_hint"] == "binary"

    def test_audit_log_appended_by_profiling(self, clean_df):
        result = self._run(clean_df)
        steps = [e["step"] for e in result["audit_log"]]
        assert "profiling" in steps

    def test_profiling_audit_status_ok(self, clean_df):
        result = self._run(clean_df)
        profiling_entry = next(e for e in result["audit_log"] if e["step"] == "profiling")
        assert profiling_entry["status"] == "ok"

    def test_high_missing_cols_flagged_in_audit(self, messy_df):
        result = self._run(messy_df)
        profiling_entry = next(e for e in result["audit_log"] if e["step"] == "profiling")
        # "score" col is 100% null — should be in high_missing_cols
        assert "score" in profiling_entry["high_missing_cols"]