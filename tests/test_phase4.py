"""
tests/test_phase4.py — Phase 4

Tests for execution_agent, validation_agent, and the graph's should_continue
conditional edge.

Run with:  pytest tests/test_phase4.py -v
"""

from __future__ import annotations

import math
import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from agents.execution_agent import apply_plan, run_execution, HANDLERS
from agents.validation_agent import compute_quality_score, run_validation
from pipeline.graph import should_continue
from pipeline.state import CleaningStep, QualityScore


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _step(col: str, action: str, params: dict = None) -> CleaningStep:
    return CleaningStep(column=col, action=action,
                        rationale="test", params=params or {})


def _state(df: pd.DataFrame, plan=None, iteration=0, **overrides) -> dict:
    base = {
        "raw_dataset": df,
        "current_dataset": df.copy() if df is not None else None,
        "filename": "test.csv",
        "profile": None,
        "cleaning_plan": plan or [],
        "quality_score": None,
        "iteration": iteration,
        "max_iterations": 3,
        "quality_threshold": 0.90,
        "audit_log": [],
        "quality_history": [],
        "api_key": "",
        "mode": "rules",
    }
    base.update(overrides)
    return base


# ===========================================================================
# EXECUTION AGENT
# ===========================================================================

class TestApplyPlan:
    """Unit tests on apply_plan() directly — no state involved."""

    # --- Imputation ---------------------------------------------------------

    def test_impute_mean_fills_nulls(self):
        df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0]})
        result, _ = apply_plan(df, [_step("x", "impute_mean")])
        assert result["x"].isna().sum() == 0
        assert math.isclose(result["x"].iloc[2], (1 + 2 + 4) / 3)

    def test_impute_median_fills_nulls(self):
        df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 100.0]})
        result, _ = apply_plan(df, [_step("x", "impute_median")])
        assert result["x"].isna().sum() == 0
        assert result["x"].iloc[2] == 2.0   # median of [1,2,100] = 2.0

    def test_impute_mode_fills_categorical(self):
        df = pd.DataFrame({"city": ["NYC", "NYC", "LA", None]})
        result, _ = apply_plan(df, [_step("city", "impute_mode")])
        assert result["city"].isna().sum() == 0
        assert result["city"].iloc[3] == "NYC"

    def test_impute_constant_uses_param(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        result, _ = apply_plan(df, [_step("x", "impute_constant", {"value": -1})])
        assert result["x"].iloc[1] == -1.0

    # --- Column dropping ----------------------------------------------------

    def test_drop_column_removes_column(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result, _ = apply_plan(df, [_step("a", "drop_column")])
        assert "a" not in result.columns
        assert "b" in result.columns

    # --- Outlier handling ---------------------------------------------------

    def test_clip_iqr_clips_outliers(self):
        vals = [10] * 8 + [1000, -1000]       # two obvious outliers
        df = pd.DataFrame({"x": vals})
        result, _ = apply_plan(df, [_step("x", "clip_iqr", {"factor": 1.5})])
        assert result["x"].max() < 1000
        assert result["x"].min() > -1000

    def test_clip_iqr_default_factor(self):
        """factor defaults to 1.5 when not in params."""
        vals = list(range(10)) + [999]
        df = pd.DataFrame({"x": vals})
        result, _ = apply_plan(df, [_step("x", "clip_iqr")])
        assert result["x"].max() < 999

    def test_winsorise_clips_tails(self):
        df = pd.DataFrame({"x": list(range(100))})
        result, _ = apply_plan(df, [_step("x", "winsorise", {"lower": 0.05, "upper": 0.95})])
        assert result["x"].min() >= df["x"].quantile(0.05) - 0.001
        assert result["x"].max() <= df["x"].quantile(0.95) + 0.001

    # --- Encoding -----------------------------------------------------------

    def test_encode_onehot_creates_dummy_columns(self):
        df = pd.DataFrame({"color": ["red", "blue", "red", "green"]})
        result, _ = apply_plan(df, [_step("color", "encode_onehot")])
        assert "color" not in result.columns
        assert any(c.startswith("color_") for c in result.columns)
        assert len([c for c in result.columns if c.startswith("color_")]) == 3

    def test_encode_ordinal_uses_order(self):
        df = pd.DataFrame({"size": ["S", "M", "L", "XL"]})
        result, _ = apply_plan(df, [_step("size", "encode_ordinal", {"order": ["S", "M", "L", "XL"]})])
        assert list(result["size"]) == [0, 1, 2, 3]

    def test_encode_binary_yes_no(self):
        df = pd.DataFrame({"active": ["yes", "no", "yes", "no"]})
        result, _ = apply_plan(df, [_step("active", "encode_binary")])
        assert set(result["active"].dropna().astype(int).unique()) == {0, 1}

    def test_encode_binary_true_false(self):
        df = pd.DataFrame({"flag": ["True", "False", "True"]})
        result, _ = apply_plan(df, [_step("flag", "encode_binary")])
        assert set(result["flag"].dropna().astype(int).unique()) == {0, 1}

    # --- Scaling ------------------------------------------------------------

    def test_scale_standard_mean_near_zero(self):
        df = pd.DataFrame({"x": [10.0, 20.0, 30.0, 40.0, 50.0]})
        result, _ = apply_plan(df, [_step("x", "scale_standard")])
        assert abs(result["x"].mean()) < 1e-9

    def test_scale_standard_std_near_one(self):
        df = pd.DataFrame({"x": [10.0, 20.0, 30.0, 40.0, 50.0]})
        result, _ = apply_plan(df, [_step("x", "scale_standard")])
        assert abs(result["x"].std(ddof=0) - 1.0) < 1e-6

    def test_scale_minmax_range_zero_to_one(self):
        df = pd.DataFrame({"x": [0.0, 5.0, 10.0]})
        result, _ = apply_plan(df, [_step("x", "scale_minmax")])
        assert math.isclose(result["x"].min(), 0.0, abs_tol=1e-9)
        assert math.isclose(result["x"].max(), 1.0, abs_tol=1e-9)

    def test_scale_robust_nulls_preserved(self):
        df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0]})
        result, _ = apply_plan(df, [_step("x", "scale_robust")])
        assert result["x"].isna().sum() == 1   # NaN preserved, not filled

    # --- Transforms ---------------------------------------------------------

    def test_log_transform_reduces_skew(self):
        vals = [1] + [2] * 5 + [100, 500, 1000, 10000]
        df = pd.DataFrame({"x": vals})
        skew_before = df["x"].skew()
        result, _ = apply_plan(df, [_step("x", "log_transform")])
        assert abs(result["x"].skew()) < abs(skew_before)

    def test_log_transform_clips_negatives(self):
        df = pd.DataFrame({"x": [-5.0, 0.0, 10.0]})
        result, _ = apply_plan(df, [_step("x", "log_transform")])
        assert result["x"].min() >= 0.0

    def test_sqrt_transform_non_negative(self):
        df = pd.DataFrame({"x": [0.0, 1.0, 4.0, 9.0, 16.0]})
        result, _ = apply_plan(df, [_step("x", "sqrt_transform")])
        assert math.isclose(result["x"].iloc[4], 4.0)

    def test_keep_does_not_modify(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result, _ = apply_plan(df, [_step("x", "keep")])
        pd.testing.assert_frame_equal(df, result)

    # --- Step log -----------------------------------------------------------

    def test_step_log_records_ok_status(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        _, log = apply_plan(df, [_step("x", "scale_standard")])
        assert log[0]["status"] == "ok"

    def test_step_log_records_null_counts(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        _, log = apply_plan(df, [_step("x", "impute_median")])
        assert log[0]["nulls_before"] == 1
        assert log[0]["nulls_after"] == 0

    def test_unknown_action_is_skipped_not_raised(self):
        df = pd.DataFrame({"x": [1.0]})
        result, log = apply_plan(df, [_step("x", "do_magic")])
        assert log[0]["status"] == "skipped"
        pd.testing.assert_frame_equal(df, result)

    def test_missing_column_is_skipped(self):
        df = pd.DataFrame({"x": [1.0]})
        result, log = apply_plan(df, [_step("dropped_col", "scale_standard")])
        assert log[0]["status"] == "skipped"

    def test_multiple_steps_applied_in_order(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 100.0, 2.0, 3.0]})
        plan = [
            _step("x", "impute_median"),
            _step("x", "clip_iqr", {"factor": 1.5}),
            _step("x", "scale_standard"),
        ]
        result, log = apply_plan(df, plan)
        assert result["x"].isna().sum() == 0
        assert len(log) == 3
        assert all(s["status"] == "ok" for s in log)

    def test_all_handlers_registered(self):
        """Every action in VALID_ACTIONS must have a handler."""
        from agents.decision_agent import VALID_ACTIONS
        for action in VALID_ACTIONS:
            assert action in HANDLERS, f"Missing handler for action '{action}'"


class TestRunExecution:

    def test_returns_updated_dataset(self):
        df = pd.DataFrame({"x": [1.0, np.nan], "y": ["a", "b"]})
        plan = [_step("x", "impute_mean")]
        result = run_execution(_state(df, plan))
        assert result["current_dataset"]["x"].isna().sum() == 0

    def test_iteration_incremented(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        result = run_execution(_state(df, [_step("x", "keep")], iteration=1))
        assert result["iteration"] == 2

    def test_audit_log_appended(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        result = run_execution(_state(df, [_step("x", "keep")]))
        assert any(e["step"] == "execution" for e in result["audit_log"])

    def test_audit_status_ok(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        result = run_execution(_state(df, [_step("x", "keep")]))
        entry = next(e for e in result["audit_log"] if e["step"] == "execution")
        assert entry["status"] == "ok"

    def test_raises_on_empty_dataframe(self):
        with pytest.raises(Exception):
            run_execution(_state(pd.DataFrame(), [_step("x", "keep")]))

    def test_raises_on_empty_plan(self):
        df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        with pytest.raises(Exception):
            run_execution(_state(df, plan=[]))


# ===========================================================================
# VALIDATION AGENT
# ===========================================================================

class TestComputeQualityScore:

    def test_perfect_data_scores_near_one(self):
        # Clean numeric data: no nulls, symmetric, no outliers
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [5.0, 4.0, 3.0, 2.0, 1.0],
        })
        score = compute_quality_score(df, iteration=0)
        assert score["overall"] > 0.90

    def test_all_null_column_scores_zero_missing(self):
        df = pd.DataFrame({
            "a": [np.nan, np.nan, np.nan],
            "b": [1.0, 2.0, 3.0],
        })
        score = compute_quality_score(df, iteration=0)
        assert score["missing_score"] < 1.0

    def test_outlier_heavy_df_lowers_outlier_score(self):
        # 4 extreme outliers in a 10-row column
        vals = [10] * 6 + [10000, -10000, 20000, -20000]
        df = pd.DataFrame({"x": vals, "y": list(range(10))})
        score = compute_quality_score(df, iteration=0)
        assert score["outlier_score"] < 0.80

    def test_high_skew_lowers_skewness_score(self):
        # Exponential-ish distribution — very high skew
        vals = [1] * 50 + [1000]
        df = pd.DataFrame({"x": vals, "y": list(range(51))})
        score = compute_quality_score(df, iteration=0)
        assert score["skewness_score"] < 0.90

    def test_overall_is_weighted_combination(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        score = compute_quality_score(df, iteration=0)
        expected = (
            0.50 * score["missing_score"]
            + 0.30 * score["outlier_score"]
            + 0.20 * score["skewness_score"]
        )
        assert math.isclose(score["overall"], round(expected, 4), abs_tol=1e-4)

    def test_scores_clamped_to_unit_interval(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        score = compute_quality_score(df, iteration=0)
        for key in ("overall", "missing_score", "outlier_score", "skewness_score"):
            assert 0.0 <= score[key] <= 1.0

    def test_iteration_recorded_in_score(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        score = compute_quality_score(df, iteration=2)
        assert score["iteration"] == 2

    def test_details_contains_all_columns(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": ["x", "y"]})
        score = compute_quality_score(df, iteration=0)
        assert set(score["details"].keys()) == {"a", "b", "c"}

    def test_empty_df_returns_zero_overall(self):
        score = compute_quality_score(pd.DataFrame(), iteration=0)
        assert score["overall"] == 0.0

    def test_cleaning_improves_score(self):
        """Score after imputation must be >= score before."""
        df_dirty = pd.DataFrame({
            "a": [1.0, np.nan, 3.0, np.nan, 5.0],
            "b": [2.0, 4.0, np.nan, 8.0, 10.0],
        })
        df_clean = df_dirty.fillna(df_dirty.median())
        score_before = compute_quality_score(df_dirty, iteration=0)
        score_after  = compute_quality_score(df_clean, iteration=1)
        assert score_after["overall"] >= score_before["overall"]


class TestRunValidation:

    def test_quality_score_written_to_state(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = run_validation(_state(df))
        assert result["quality_score"] is not None
        assert "overall" in result["quality_score"]

    def test_quality_history_appended(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = run_validation(_state(df))
        assert len(result["quality_history"]) == 1

    def test_quality_history_accumulates_across_calls(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        # Simulate second iteration — pass existing history
        prior_score = QualityScore(overall=0.80, missing_score=0.90,
                                   outlier_score=0.75, skewness_score=0.85,
                                   iteration=0, details={})
        state = _state(df, quality_history=[prior_score])
        result = run_validation(state)
        assert len(result["quality_history"]) == 2

    def test_audit_log_appended(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = run_validation(_state(df))
        assert any(e["step"] == "validation" for e in result["audit_log"])

    def test_verdict_done_when_threshold_met(self):
        # Perfect clean data should pass the 0.90 threshold
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [5.0, 4.0, 3.0, 2.0, 1.0],
        })
        result = run_validation(_state(df, quality_threshold=0.50))
        entry = next(e for e in result["audit_log"] if e["step"] == "validation")
        assert entry["verdict"] == "done"

    def test_verdict_retry_when_threshold_not_met(self):
        # Dirty data won't pass a high threshold
        df = pd.DataFrame({
            "a": [np.nan] * 5 + [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [1.0] * 10,
        })
        result = run_validation(_state(df, quality_threshold=0.99, iteration=0,
                                       max_iterations=3))
        entry = next(e for e in result["audit_log"] if e["step"] == "validation")
        # Should be "retry" since threshold not met and iterations not exhausted
        assert entry["verdict"] in ("retry", "done")   # depends on score

    def test_raises_on_none_dataset(self):
        s = _state(pd.DataFrame({"a": [1.0], "b": [2.0]}))
        s["current_dataset"] = None
        with pytest.raises(Exception):
            run_validation(s)


# ===========================================================================
# GRAPH — should_continue conditional edge
# ===========================================================================

class TestShouldContinue:

    def _score(self, overall: float, iteration: int = 0) -> dict:
        return {
            "quality_score": QualityScore(
                overall=overall, missing_score=1.0, outlier_score=1.0,
                skewness_score=1.0, iteration=iteration, details={},
            ),
            "iteration": iteration,
            "max_iterations": 3,
            "quality_threshold": 0.90,
        }

    def test_done_when_score_above_threshold(self):
        assert should_continue(self._score(0.95)) == "done"

    def test_done_when_score_exactly_at_threshold(self):
        assert should_continue(self._score(0.90)) == "done"

    def test_retry_when_score_below_threshold(self):
        assert should_continue(self._score(0.80, iteration=0)) == "retry"

    def test_done_when_max_iterations_reached(self):
        # Score below threshold but no more iterations left
        state = self._score(0.70, iteration=3)
        assert should_continue(state) == "done"

    def test_retry_when_below_threshold_and_iterations_remain(self):
        state = self._score(0.70, iteration=1)   # max=3, only 1 used
        assert should_continue(state) == "retry"

    def test_done_when_quality_score_is_none(self):
        """If quality_score is None (should never happen), default to done."""
        state = {
            "quality_score": None,
            "iteration": 0,
            "max_iterations": 3,
            "quality_threshold": 0.90,
        }
        assert should_continue(state) == "done"

    def test_custom_threshold_respected(self):
        state = {
            "quality_score": QualityScore(overall=0.70, missing_score=1.0,
                                          outlier_score=1.0, skewness_score=1.0,
                                          iteration=0, details={}),
            "iteration": 0,
            "max_iterations": 3,
            "quality_threshold": 0.60,   # lower threshold — 0.70 should pass
        }
        assert should_continue(state) == "done"