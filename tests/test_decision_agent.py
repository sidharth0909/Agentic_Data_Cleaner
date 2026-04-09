"""
tests/test_decision_agent.py  — Phase 3

Run with:  pytest tests/test_decision_agent.py -v
"""

from __future__ import annotations

import json
import sys
import os
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from agents.decision_agent import (
    VALID_ACTIONS,
    build_rules_plan,
    run_decision,
    _rules_decision,
    _validate_and_coerce_plan,
)
from pipeline.state import ColumnProfile, CleaningStep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _profile(
    column: str = "x",
    dtype: str = "float64",
    missing_pct: float = 0.0,
    unique_count: int = 50,
    cardinality_hint: str = "medium",
    skewness: float | None = 0.1,
    outlier_ratio: float | None = 0.0,
    sample_values: list = None,
    numeric_summary: dict | None = None,
) -> ColumnProfile:
    return ColumnProfile(
        column=column,
        dtype=dtype,
        missing_pct=missing_pct,
        unique_count=unique_count,
        cardinality_hint=cardinality_hint,
        skewness=skewness,
        outlier_ratio=outlier_ratio,
        sample_values=sample_values or [1.0, 2.0, 3.0],
        numeric_summary=numeric_summary or {"mean": 5.0, "std": 1.0, "min": 0.0, "max": 10.0},
    )


def _state(profiles: list[ColumnProfile], mode: str = "rules", **overrides) -> dict:
    base = {
        "raw_dataset": None,
        "current_dataset": None,
        "filename": "test.csv",
        "profile": profiles,
        "cleaning_plan": None,
        "quality_score": None,
        "iteration": 0,
        "max_iterations": 3,
        "quality_threshold": 0.90,
        "audit_log": [],
        "quality_history": [],
        "api_key": "sk-test-key",
        "mode": mode,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Rules engine — unit tests on _rules_decision
# ---------------------------------------------------------------------------

class TestRulesDecision:

    def test_fully_null_column_is_dropped(self):
        p = _profile(missing_pct=1.0)
        step = _rules_decision(p)
        assert step["action"] == "drop_column"

    def test_identifier_column_is_kept(self):
        p = _profile(cardinality_hint="identifier")
        step = _rules_decision(p)
        assert step["action"] == "keep"

    def test_numeric_missing_gets_impute_median(self):
        p = _profile(missing_pct=0.25)          # numeric (has skewness)
        step = _rules_decision(p)
        assert step["action"] == "impute_median"

    def test_categorical_missing_gets_impute_mode(self):
        p = _profile(
            dtype="object",
            skewness=None,          # object column → no skewness
            outlier_ratio=None,
            missing_pct=0.10,
            cardinality_hint="low",
            numeric_summary=None,
        )
        step = _rules_decision(p)
        assert step["action"] == "impute_mode"

    def test_high_outlier_and_high_skew_nonneg_gets_log_transform(self):
        p = _profile(
            skewness=2.5,
            outlier_ratio=0.12,
            numeric_summary={"mean": 5.0, "std": 2.0, "min": 0.0, "max": 100.0},
        )
        step = _rules_decision(p)
        assert step["action"] == "log_transform"

    def test_high_outlier_and_high_skew_with_negatives_gets_clip(self):
        p = _profile(
            skewness=2.5,
            outlier_ratio=0.12,
            numeric_summary={"mean": 0.0, "std": 5.0, "min": -10.0, "max": 100.0},
        )
        step = _rules_decision(p)
        assert step["action"] == "clip_iqr"

    def test_high_outlier_only_gets_clip_iqr(self):
        p = _profile(skewness=0.3, outlier_ratio=0.08)
        step = _rules_decision(p)
        assert step["action"] == "clip_iqr"
        assert step["params"].get("factor") == 1.5

    def test_high_skew_only_nonneg_gets_log_transform(self):
        p = _profile(
            skewness=1.8,
            outlier_ratio=0.01,
            numeric_summary={"mean": 5.0, "std": 2.0, "min": 0.0, "max": 50.0},
        )
        step = _rules_decision(p)
        assert step["action"] == "log_transform"

    def test_categorical_low_cardinality_gets_onehot(self):
        p = _profile(
            dtype="object",
            skewness=None,
            outlier_ratio=None,
            missing_pct=0.0,
            cardinality_hint="low",
            numeric_summary=None,
        )
        step = _rules_decision(p)
        assert step["action"] == "encode_onehot"

    def test_numeric_no_issues_gets_scale_standard(self):
        p = _profile(skewness=0.2, outlier_ratio=0.0)
        step = _rules_decision(p)
        assert step["action"] == "scale_standard"

    def test_numeric_minor_outliers_gets_scale_robust(self):
        p = _profile(skewness=0.2, outlier_ratio=0.03)
        step = _rules_decision(p)
        assert step["action"] == "scale_robust"

    def test_all_actions_are_valid(self):
        """Every action emitted by the rules engine must be in VALID_ACTIONS."""
        scenarios = [
            _profile(missing_pct=1.0),
            _profile(cardinality_hint="identifier"),
            _profile(missing_pct=0.3),
            _profile(dtype="object", skewness=None, outlier_ratio=None,
                     missing_pct=0.1, cardinality_hint="low", numeric_summary=None),
            _profile(skewness=2.5, outlier_ratio=0.12,
                     numeric_summary={"mean": 5.0, "std": 2.0, "min": 0.0, "max": 100.0}),
            _profile(skewness=0.3, outlier_ratio=0.08),
            _profile(skewness=1.8, outlier_ratio=0.01,
                     numeric_summary={"mean": 5.0, "std": 2.0, "min": 0.0, "max": 50.0}),
            _profile(dtype="object", skewness=None, outlier_ratio=None,
                     missing_pct=0.0, cardinality_hint="low", numeric_summary=None),
            _profile(skewness=0.2, outlier_ratio=0.0),
        ]
        for p in scenarios:
            step = _rules_decision(p)
            assert step["action"] in VALID_ACTIONS, (
                f"Action '{step['action']}' for column '{p['column']}' is not in VALID_ACTIONS"
            )

    def test_rationale_is_nonempty_string(self):
        p = _profile(skewness=0.2, outlier_ratio=0.0)
        step = _rules_decision(p)
        assert isinstance(step["rationale"], str)
        assert len(step["rationale"]) > 10

    def test_step_has_required_keys(self):
        p = _profile()
        step = _rules_decision(p)
        for key in ("column", "action", "rationale", "params"):
            assert key in step


# ---------------------------------------------------------------------------
# Rules engine — integration via build_rules_plan
# ---------------------------------------------------------------------------

class TestBuildRulesPlan:

    def test_one_step_per_column(self):
        profiles = [
            _profile(column="age"),
            _profile(column="income"),
            _profile(column="city", dtype="object", skewness=None,
                     outlier_ratio=None, numeric_summary=None, cardinality_hint="low"),
        ]
        plan = build_rules_plan(profiles)
        assert len(plan) == 3
        cols = [s["column"] for s in plan]
        assert set(cols) == {"age", "income", "city"}


# ---------------------------------------------------------------------------
# Plan validation / coercion
# ---------------------------------------------------------------------------

class TestValidateAndCoercePlan:

    def _profiles(self, cols: list[str]) -> list[ColumnProfile]:
        return [_profile(column=c) for c in cols]

    def test_valid_plan_passes_through(self):
        profiles = self._profiles(["a", "b"])
        raw = [
            {"column": "a", "action": "scale_standard", "rationale": "ok", "params": {}},
            {"column": "b", "action": "impute_median", "rationale": "ok", "params": {}},
        ]
        plan = _validate_and_coerce_plan(raw, profiles)
        assert len(plan) == 2
        assert plan[0]["action"] == "scale_standard"

    def test_unknown_action_replaced_with_keep(self):
        profiles = self._profiles(["a"])
        raw = [{"column": "a", "action": "do_magic", "rationale": "x", "params": {}}]
        plan = _validate_and_coerce_plan(raw, profiles)
        assert plan[0]["action"] == "keep"

    def test_missing_column_gets_keep_step(self):
        profiles = self._profiles(["a", "b"])
        raw = [{"column": "a", "action": "scale_standard", "rationale": "x", "params": {}}]
        plan = _validate_and_coerce_plan(raw, profiles)
        cols = {s["column"] for s in plan}
        assert "b" in cols
        b_step = next(s for s in plan if s["column"] == "b")
        assert b_step["action"] == "keep"

    def test_extra_unknown_column_is_dropped(self):
        profiles = self._profiles(["a"])
        raw = [
            {"column": "a", "action": "keep", "rationale": "x", "params": {}},
            {"column": "z_unknown", "action": "keep", "rationale": "x", "params": {}},
        ]
        plan = _validate_and_coerce_plan(raw, profiles)
        cols = {s["column"] for s in plan}
        assert "z_unknown" not in cols

    def test_null_params_normalised_to_empty_dict(self):
        profiles = self._profiles(["a"])
        raw = [{"column": "a", "action": "keep", "rationale": "x", "params": None}]
        plan = _validate_and_coerce_plan(raw, profiles)
        assert plan[0]["params"] == {}


# ---------------------------------------------------------------------------
# run_decision node — rules mode
# ---------------------------------------------------------------------------

class TestRunDecisionRules:

    def test_returns_cleaning_plan(self):
        profiles = [_profile(column="age"), _profile(column="income")]
        result = run_decision(_state(profiles, mode="rules"))
        assert "cleaning_plan" in result
        assert len(result["cleaning_plan"]) == 2

    def test_audit_log_appended(self):
        result = run_decision(_state([_profile()], mode="rules"))
        assert any(e["step"] == "decision" for e in result["audit_log"])

    def test_audit_status_ok(self):
        result = run_decision(_state([_profile()], mode="rules"))
        entry = next(e for e in result["audit_log"] if e["step"] == "decision")
        assert entry["status"] == "ok"

    def test_audit_plan_source_is_rules(self):
        result = run_decision(_state([_profile()], mode="rules"))
        entry = next(e for e in result["audit_log"] if e["step"] == "decision")
        assert entry["plan_source"] == "rules"

    def test_action_summary_in_audit(self):
        profiles = [_profile(column="a"), _profile(column="b")]
        result = run_decision(_state(profiles, mode="rules"))
        entry = next(e for e in result["audit_log"] if e["step"] == "decision")
        assert "action_summary" in entry

    def test_raises_on_empty_profile(self):
        with pytest.raises(Exception):
            run_decision(_state([], mode="rules"))

    def test_existing_audit_log_preserved(self):
        prior = [{"step": "profiling", "status": "ok"}]
        result = run_decision(_state([_profile()], mode="rules", audit_log=prior))
        steps = [e["step"] for e in result["audit_log"]]
        assert "profiling" in steps
        assert "decision" in steps

    def test_all_plan_actions_valid(self):
        profiles = [
            _profile(column="a", missing_pct=0.3),
            _profile(column="b", skewness=2.5, outlier_ratio=0.15,
                     numeric_summary={"mean": 5.0, "std": 2.0, "min": 0.0, "max": 100.0}),
            _profile(column="c", dtype="object", skewness=None, outlier_ratio=None,
                     cardinality_hint="low", numeric_summary=None),
        ]
        result = run_decision(_state(profiles, mode="rules"))
        for step in result["cleaning_plan"]:
            assert step["action"] in VALID_ACTIONS


# ---------------------------------------------------------------------------
# run_decision node — LLM mode (mocked)
# ---------------------------------------------------------------------------

class TestRunDecisionLLM:
    """
    We mock langchain_anthropic.ChatAnthropic so no real API call is made.
    Tests verify that run_decision correctly wires the LLM path: passes the
    prompt, parses the response, and populates the plan and audit log.
    """

    def _mock_response(self, plan: list[dict]) -> MagicMock:
        msg = MagicMock()
        msg.content = json.dumps(plan)
        return msg

    def test_llm_plan_used_when_api_key_present(self):
        llm_plan = [
            {"column": "age", "action": "impute_median",
             "rationale": "Missing values in age.", "params": {}},
            {"column": "income", "action": "scale_robust",
             "rationale": "Income has outliers.", "params": {}},
        ]
        profiles = [_profile(column="age"), _profile(column="income")]

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._mock_response(llm_plan)

        with patch("agents.decision_agent.ChatAnthropic", return_value=mock_llm):
            with patch("agents.decision_agent.SystemMessage", side_effect=lambda **kw: kw):
                with patch("agents.decision_agent.HumanMessage", side_effect=lambda **kw: kw):
                    result = run_decision(_state(profiles, mode="llm", api_key="sk-real"))

        assert result["cleaning_plan"][0]["action"] == "impute_median"
        entry = next(e for e in result["audit_log"] if e["step"] == "decision")
        assert entry["plan_source"] == "llm"

    def test_fallback_to_rules_when_no_api_key(self):
        profiles = [_profile(column="age")]
        result = run_decision(_state(profiles, mode="llm", api_key=""))
        entry = next(e for e in result["audit_log"] if e["step"] == "decision")
        assert entry["plan_source"] == "rules_fallback_no_key"

    def test_fallback_to_rules_on_llm_exception(self):
        profiles = [_profile(column="age")]
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("API timeout")

        with patch("agents.decision_agent.ChatAnthropic", return_value=mock_llm):
            with patch("agents.decision_agent.SystemMessage", side_effect=lambda **kw: kw):
                with patch("agents.decision_agent.HumanMessage", side_effect=lambda **kw: kw):
                    result = run_decision(_state(profiles, mode="llm", api_key="sk-real"))

        entry = next(e for e in result["audit_log"] if e["step"] == "decision")
        assert entry["plan_source"] == "rules_fallback_llm_error"
        assert "llm_error" in entry

    def test_llm_json_with_fences_is_parsed(self):
        """Verify that accidental markdown code fences are stripped."""
        llm_plan = [
            {"column": "age", "action": "keep", "rationale": "Fine.", "params": {}},
        ]
        fenced_response = f"```json\n{json.dumps(llm_plan)}\n```"
        profiles = [_profile(column="age")]

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=fenced_response)

        with patch("agents.decision_agent.ChatAnthropic", return_value=mock_llm):
            with patch("agents.decision_agent.SystemMessage", side_effect=lambda **kw: kw):
                with patch("agents.decision_agent.HumanMessage", side_effect=lambda **kw: kw):
                    result = run_decision(_state(profiles, mode="llm", api_key="sk-real"))

        assert result["cleaning_plan"][0]["action"] == "keep"

    def test_iteration_passed_to_prompt(self):
        """Verify previous_plan is included when iteration > 0."""
        prior_plan = [
            CleaningStep(column="age", action="impute_median",
                         rationale="First pass.", params={})
        ]
        profiles = [_profile(column="age")]

        captured_messages = []

        def capture_invoke(messages):
            captured_messages.extend(messages)
            plan = [{"column": "age", "action": "scale_standard",
                     "rationale": "Second pass.", "params": {}}]
            return MagicMock(content=json.dumps(plan))

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = capture_invoke

        with patch("agents.decision_agent.ChatAnthropic", return_value=mock_llm):
            with patch("agents.decision_agent.SystemMessage", side_effect=lambda **kw: kw):
                with patch("agents.decision_agent.HumanMessage", side_effect=lambda **kw: kw):
                    run_decision(_state(
                        profiles, mode="llm", api_key="sk-real",
                        iteration=1, cleaning_plan=prior_plan,
                    ))

        # The human message content should mention iteration 1 and the previous plan
        human_msg = captured_messages[-1]
        content = human_msg.get("content", "")
        assert "Iteration: 1" in content
        assert "impute_median" in content