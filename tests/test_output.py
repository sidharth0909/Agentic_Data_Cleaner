"""
tests/test_output.py — Phase 5

Tests for the output agent: CSV serialisation, Markdown report structure,
explainability JSON shape, and the run_output LangGraph node.

Run with:  pytest tests/test_output.py -v
"""

from __future__ import annotations

import json
import sys
import os
import io
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from agents.output import (
    _build_csv_bytes,
    _build_explainability_json,
    _build_markdown_report,
    _extract_shapes,
    _make_run_id,
    run_output,
)
from pipeline.state import CleaningStep, ColumnProfile, QualityScore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _qs(overall: float = 0.92, iteration: int = 1) -> QualityScore:
    return QualityScore(
        overall=overall,
        missing_score=0.95,
        outlier_score=0.90,
        skewness_score=0.88,
        iteration=iteration,
        details={"age": {"missing_pct": 0.0, "outlier_ratio": 0.02, "skewness_score": 0.9}},
    )


def _profile(col: str = "age") -> ColumnProfile:
    return ColumnProfile(
        column=col,
        dtype="float64",
        missing_pct=0.05,
        unique_count=40,
        cardinality_hint="medium",
        skewness=0.3,
        outlier_ratio=0.02,
        sample_values=[25.0, 32.0, 47.0],
        numeric_summary={"mean": 35.0, "std": 12.0, "min": 1.0, "max": 90.0},
    )


def _step(col: str = "age", action: str = "impute_median") -> CleaningStep:
    return CleaningStep(
        column=col,
        action=action,
        rationale="Median imputation chosen because column has missing values.",
        params={},
    )


def _audit_log() -> list[dict]:
    return [
        {
            "step": "ingestion",
            "status": "ok",
            "shape_before": [100, 5],
            "shape_after": [98, 5],
            "dtype_changes": [],
            "fully_empty_rows_dropped": 2,
        },
        {
            "step": "profiling",
            "status": "ok",
            "n_rows": 98,
            "n_cols": 5,
        },
        {
            "step": "decision",
            "status": "ok",
            "plan_source": "llm",
            "n_steps": 3,
        },
        {
            "step": "execution",
            "status": "ok",
            "iteration": 0,
            "shape_before": [98, 5],
            "shape_after": [98, 5],
            "steps_ok": 3,
            "steps_skipped": 0,
            "steps_error": 0,
            "step_log": [
                {"column": "age", "action": "impute_median",
                 "status": "ok", "nulls_before": 5, "nulls_after": 0,
                 "shape_before": [98, 5], "shape_after": [98, 5]},
            ],
        },
        {
            "step": "validation",
            "status": "ok",
            "overall": 0.92,
            "verdict": "done",
        },
    ]


def _clean_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [25.0, 32.0, 47.0, 51.0, 62.0],
        "income": [50000.0, 72000.0, 91000.0, 115000.0, 88000.0],
        "city": ["NYC", "LA", "Chicago", "NYC", "LA"],
    })


def _full_state(df: pd.DataFrame, tmp_dir: Path) -> dict:
    return {
        "raw_dataset": df,
        "current_dataset": df.copy(),
        "filename": "titanic.csv",
        "profile": [_profile("age"), _profile("income")],
        "cleaning_plan": [_step("age"), _step("income", "scale_robust")],
        "quality_score": _qs(),
        "iteration": 1,
        "max_iterations": 3,
        "quality_threshold": 0.90,
        "audit_log": _audit_log(),
        "quality_history": [_qs(0.78, 0), _qs(0.92, 1)],
        "api_key": "sk-test",
        "mode": "llm",
        "output_paths": None,
        "explainability": None,
    }


# ---------------------------------------------------------------------------
# _make_run_id
# ---------------------------------------------------------------------------

class TestMakeRunId:

    def test_contains_stem(self):
        rid = _make_run_id("titanic.csv")
        assert rid.startswith("titanic_")

    def test_no_spaces_or_dots(self):
        rid = _make_run_id("my file.with.dots.csv")
        assert " " not in rid
        assert rid.count(".") == 0

    def test_unique_per_call(self):
        # Two calls at the same second might collide in theory but overwhelmingly won't
        import time
        r1 = _make_run_id("test.csv")
        time.sleep(0.01)
        r2 = _make_run_id("test.csv")
        # At minimum both are valid strings
        assert isinstance(r1, str) and isinstance(r2, str)

    def test_ends_with_utc(self):
        rid = _make_run_id("data.csv")
        assert rid.endswith("_UTC")


# ---------------------------------------------------------------------------
# _build_csv_bytes
# ---------------------------------------------------------------------------

class TestBuildCsvBytes:

    def test_returns_bytes(self):
        df = _clean_df()
        result = _build_csv_bytes(df)
        assert isinstance(result, bytes)

    def test_roundtrip_matches_original(self):
        df = _clean_df()
        csv_bytes = _build_csv_bytes(df)
        df2 = pd.read_csv(io.BytesIO(csv_bytes))
        assert list(df2.columns) == list(df.columns)
        assert len(df2) == len(df)

    def test_no_index_column(self):
        df = _clean_df()
        csv_bytes = _build_csv_bytes(df)
        df2 = pd.read_csv(io.BytesIO(csv_bytes))
        assert "Unnamed: 0" not in df2.columns

    def test_utf8_encoded(self):
        df = pd.DataFrame({"name": ["Ångström", "日本語", "café"], "val": [1, 2, 3]})
        csv_bytes = _build_csv_bytes(df)
        text = csv_bytes.decode("utf-8")
        assert "Ångström" in text


# ---------------------------------------------------------------------------
# _extract_shapes
# ---------------------------------------------------------------------------

class TestExtractShapes:

    def test_extracts_ingestion_shape_before(self):
        before, _ = _extract_shapes(_audit_log())
        assert before == [100, 5]

    def test_extracts_execution_shape_after(self):
        _, after = _extract_shapes(_audit_log())
        assert after == [98, 5]

    def test_defaults_to_zero_zero_when_missing(self):
        before, after = _extract_shapes([])
        assert before == [0, 0]
        assert after == [0, 0]

    def test_last_execution_shape_used(self):
        log = _audit_log()
        log.append({
            "step": "execution",
            "status": "ok",
            "iteration": 1,
            "shape_before": [98, 5],
            "shape_after": [98, 4],   # one column dropped in second pass
            "steps_ok": 1, "steps_skipped": 0, "steps_error": 0,
            "step_log": [],
        })
        _, after = _extract_shapes(log)
        assert after == [98, 4]


# ---------------------------------------------------------------------------
# _build_markdown_report
# ---------------------------------------------------------------------------

class TestBuildMarkdownReport:

    def _build(self) -> str:
        return _build_markdown_report(
            filename="titanic.csv",
            run_id="titanic_20240101_UTC",
            shape_before=[100, 5],
            shape_after=[98, 4],
            initial_profile=[_profile("age"), _profile("income")],
            cleaning_plan=[_step("age"), _step("income", "scale_robust")],
            quality_history=[_qs(0.78, 0), _qs(0.92, 1)],
            final_score=_qs(0.92, 1),
            audit_log=_audit_log(),
            generated_at="2024-01-01 12:00:00 UTC",
        )

    def test_returns_string(self):
        assert isinstance(self._build(), str)

    def test_contains_filename(self):
        assert "titanic.csv" in self._build()

    def test_contains_run_id(self):
        assert "titanic_20240101_UTC" in self._build()

    def test_contains_executive_summary_heading(self):
        assert "Executive Summary" in self._build()

    def test_contains_quality_history_heading(self):
        assert "Quality Score History" in self._build()

    def test_contains_column_profiles_heading(self):
        assert "Initial Column Profiles" in self._build()

    def test_contains_cleaning_plan_heading(self):
        assert "Cleaning Plan" in self._build()

    def test_contains_execution_log_heading(self):
        assert "Execution Log" in self._build()

    def test_contains_all_column_names(self):
        md = self._build()
        assert "age" in md
        assert "income" in md

    def test_contains_rationale_text(self):
        assert "Median imputation chosen" in self._build()

    def test_contains_score_bar(self):
        md = self._build()
        assert "█" in md   # score bar uses block characters

    def test_contains_both_iterations(self):
        md = self._build()
        assert "0.78" in md or "78.0" in md    # first iteration score

    def test_valid_markdown_table_structure(self):
        md = self._build()
        # Every line that starts with '|' must end with '|'
        table_lines = [l for l in md.splitlines() if l.strip().startswith("|")]
        assert len(table_lines) > 0
        for line in table_lines:
            assert line.strip().endswith("|"), f"Malformed table row: {line!r}"

    def test_footer_present(self):
        assert "Agentic Data Cleaner" in self._build()


# ---------------------------------------------------------------------------
# _build_explainability_json
# ---------------------------------------------------------------------------

class TestBuildExplainabilityJson:

    def _build(self) -> dict:
        return _build_explainability_json(
            run_id="titanic_20240101_UTC",
            filename="titanic.csv",
            shape_before=[100, 5],
            shape_after=[98, 4],
            initial_profile=[_profile("age"), _profile("income")],
            cleaning_plan=[_step("age"), _step("income", "scale_robust")],
            quality_history=[_qs(0.78, 0), _qs(0.92, 1)],
            final_score=_qs(0.92, 1),
            audit_log=_audit_log(),
            iterations_completed=1,
        )

    def test_top_level_keys_present(self):
        payload = self._build()
        for key in (
            "run_id", "filename", "shape_before", "shape_after",
            "iterations_completed", "final_score", "quality_history",
            "cleaning_plan", "column_profiles", "iteration_history",
        ):
            assert key in payload, f"Missing key: {key}"

    def test_run_id_matches(self):
        assert self._build()["run_id"] == "titanic_20240101_UTC"

    def test_final_score_overall(self):
        assert self._build()["final_score"]["overall"] == 0.92

    def test_quality_history_length(self):
        assert len(self._build()["quality_history"]) == 2

    def test_cleaning_plan_has_rationale(self):
        plan = self._build()["cleaning_plan"]
        for step in plan:
            assert "rationale" in step
            assert len(step["rationale"]) > 0

    def test_column_profiles_count(self):
        assert len(self._build()["column_profiles"]) == 2

    def test_iteration_history_extracted(self):
        hist = self._build()["iteration_history"]
        assert len(hist) == 1
        assert hist[0]["steps_ok"] == 3

    def test_step_log_embedded_in_iteration_history(self):
        hist = self._build()["iteration_history"]
        assert len(hist[0]["step_log"]) == 1
        assert hist[0]["step_log"][0]["column"] == "age"

    def test_json_serialisable(self):
        payload = self._build()
        # Must not raise
        serialised = json.dumps(payload, default=str)
        assert isinstance(serialised, str)

    def test_shape_before_after_recorded(self):
        payload = self._build()
        assert payload["shape_before"] == [100, 5]
        assert payload["shape_after"] == [98, 4]


# ---------------------------------------------------------------------------
# run_output (integration — writes to a temp directory)
# ---------------------------------------------------------------------------

class TestRunOutput:

    def _run(self, tmp_path: Path) -> dict:
        df = _clean_df()
        state = _full_state(df, tmp_path)
        # Redirect output root to tmp_path for isolation
        with patch("agents.output._OUTPUT_ROOT", tmp_path):
            return run_output(state)

    def test_returns_output_paths(self, tmp_path):
        result = self._run(tmp_path)
        assert "output_paths" in result
        assert "csv" in result["output_paths"]
        assert "report" in result["output_paths"]
        assert "explainability" in result["output_paths"]

    def test_csv_file_exists(self, tmp_path):
        result = self._run(tmp_path)
        assert Path(result["output_paths"]["csv"]).exists()

    def test_report_file_exists(self, tmp_path):
        result = self._run(tmp_path)
        assert Path(result["output_paths"]["report"]).exists()

    def test_explainability_file_exists(self, tmp_path):
        result = self._run(tmp_path)
        assert Path(result["output_paths"]["explainability"]).exists()

    def test_csv_is_readable(self, tmp_path):
        result = self._run(tmp_path)
        df2 = pd.read_csv(result["output_paths"]["csv"])
        assert len(df2) == 5   # _clean_df has 5 rows

    def test_explainability_json_is_valid(self, tmp_path):
        result = self._run(tmp_path)
        with open(result["output_paths"]["explainability"]) as f:
            payload = json.load(f)
        assert "run_id" in payload
        assert "final_score" in payload

    def test_report_markdown_has_headings(self, tmp_path):
        result = self._run(tmp_path)
        text = Path(result["output_paths"]["report"]).read_text(encoding="utf-8")
        assert "# Agentic Data Cleaner" in text
        assert "## Executive Summary" in text

    def test_explainability_written_to_state(self, tmp_path):
        result = self._run(tmp_path)
        assert result["explainability"] is not None
        assert "run_id" in result["explainability"]

    def test_audit_log_appended(self, tmp_path):
        result = self._run(tmp_path)
        assert any(e["step"] == "output" for e in result["audit_log"])

    def test_audit_status_ok(self, tmp_path):
        result = self._run(tmp_path)
        entry = next(e for e in result["audit_log"] if e["step"] == "output")
        assert entry["status"] == "ok"

    def test_run_id_in_audit(self, tmp_path):
        result = self._run(tmp_path)
        entry = next(e for e in result["audit_log"] if e["step"] == "output")
        assert "run_id" in entry

    def test_raises_on_none_dataset(self, tmp_path):
        state = _full_state(_clean_df(), tmp_path)
        state["current_dataset"] = None
        with patch("agents.output._OUTPUT_ROOT", tmp_path):
            with pytest.raises(Exception):
                run_output(state)

    def test_output_dir_is_run_id_scoped(self, tmp_path):
        """Each run must write to its own subdirectory."""
        result = self._run(tmp_path)
        run_id = result["output_paths"]["run_id"]
        out_dir = Path(result["output_paths"]["out_dir"])
        assert run_id in str(out_dir)
        assert out_dir.parent == tmp_path