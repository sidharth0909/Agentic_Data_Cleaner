"""
tests/test_api.py — Phase 6

Integration tests for the FastAPI endpoints.
Uses httpx.AsyncClient with the ASGI transport — no real server is started.

Run with:  pytest tests/test_api.py -v
"""

from __future__ import annotations

import io
import json
import sys
import os
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
import pytest_asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

# Import app AFTER path manipulation
from main import app

try:
    from httpx import AsyncClient, ASGITransport
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _HTTPX_AVAILABLE,
    reason="httpx not installed — run: pip install httpx",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _csv_bytes(rows: int = 20) -> bytes:
    """Generate a minimal well-formed CSV as bytes."""
    import numpy as np
    df = pd.DataFrame({
        "age":    np.random.randint(18, 80, rows).astype(float),
        "income": np.random.uniform(20000, 120000, rows),
        "city":   ["NYC", "LA", "Chicago"] * (rows // 3) + ["NYC"] * (rows % 3),
        "active": ["yes", "no"] * (rows // 2),
    })
    # Introduce a few nulls
    df.loc[0, "age"] = float("nan")
    df.loc[3, "income"] = float("nan")
    return df.to_csv(index=False).encode("utf-8")


def _upload_file(content: bytes, filename: str = "test.csv"):
    return {"file": (filename, io.BytesIO(content), "text/csv")}


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------

class TestHealth:

    @pytest.mark.asyncio
    async def test_health_returns_ok(self, client):
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# POST /api/upload
# ---------------------------------------------------------------------------

class TestUpload:

    @pytest.mark.asyncio
    async def test_valid_csv_returns_200(self, client):
        resp = await client.post(
            "/api/upload",
            files=_upload_file(_csv_bytes()),
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_response_has_session_id(self, client):
        resp = await client.post("/api/upload", files=_upload_file(_csv_bytes()))
        data = resp.json()
        assert "session_id" in data
        assert len(data["session_id"]) > 0

    @pytest.mark.asyncio
    async def test_response_has_correct_shape(self, client):
        resp = await client.post("/api/upload", files=_upload_file(_csv_bytes(20)))
        data = resp.json()
        assert data["rows"] == 20
        assert data["columns"] == 4

    @pytest.mark.asyncio
    async def test_response_has_column_names(self, client):
        resp = await client.post("/api/upload", files=_upload_file(_csv_bytes()))
        data = resp.json()
        assert "column_names" in data
        assert "age" in data["column_names"]

    @pytest.mark.asyncio
    async def test_non_csv_returns_400(self, client):
        resp = await client.post(
            "/api/upload",
            files={"file": ("data.xlsx", io.BytesIO(b"fake"), "application/octet-stream")},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_empty_file_returns_400(self, client):
        resp = await client.post(
            "/api/upload",
            files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_unparseable_csv_returns_422(self, client):
        garbage = b"\x89PNG\r\n\x1a\n"   # PNG magic bytes — not a CSV
        resp = await client.post(
            "/api/upload",
            files={"file": ("bad.csv", io.BytesIO(garbage), "text/csv")},
        )
        assert resp.status_code in (400, 422)

    @pytest.mark.asyncio
    async def test_single_column_csv_returns_422(self, client):
        csv = b"name\nAlice\nBob\n"
        resp = await client.post(
            "/api/upload",
            files={"file": ("single.csv", io.BytesIO(csv), "text/csv")},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_filename_returned(self, client):
        resp = await client.post(
            "/api/upload",
            files={"file": ("my_data.csv", io.BytesIO(_csv_bytes()), "text/csv")},
        )
        assert resp.json()["filename"] == "my_data.csv"


# ---------------------------------------------------------------------------
# GET /api/results/{run_id}
# ---------------------------------------------------------------------------

class TestGetResults:

    @pytest.mark.asyncio
    async def test_missing_run_returns_404(self, client):
        resp = await client.get("/api/results/nonexistent-run-id")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_existing_run_returns_payload(self, client, tmp_path):
        # Seed a fake explainability.json
        run_id = "test_run_123"
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        payload = {"run_id": run_id, "final_score": {"overall": 0.93}}
        (run_dir / "explainability.json").write_text(json.dumps(payload))

        with patch("main._OUTPUT_ROOT", tmp_path):
            resp = await client.get(f"/api/results/{run_id}")

        assert resp.status_code == 200
        assert resp.json()["run_id"] == run_id
        assert resp.json()["final_score"]["overall"] == 0.93


# ---------------------------------------------------------------------------
# GET /api/download/{run_id}/{file_type}
# ---------------------------------------------------------------------------

class TestDownload:

    @pytest.mark.asyncio
    async def test_missing_run_returns_404(self, client):
        resp = await client.get("/api/download/bad-run/csv")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_invalid_file_type_returns_400(self, client, tmp_path):
        run_id = "dl_test_run"
        (tmp_path / run_id).mkdir()
        with patch("main._OUTPUT_ROOT", tmp_path):
            resp = await client.get(f"/api/download/{run_id}/parquet")
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_csv_download_streams_file(self, client, tmp_path):
        run_id = "dl_csv_run"
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        csv_content = b"age,income\n25,50000\n32,72000\n"
        (run_dir / "cleaned_test.csv").write_bytes(csv_content)
        with patch("main._OUTPUT_ROOT", tmp_path):
            resp = await client.get(f"/api/download/{run_id}/csv")
        assert resp.status_code == 200
        assert b"age,income" in resp.content

    @pytest.mark.asyncio
    async def test_report_download_streams_file(self, client, tmp_path):
        run_id = "dl_report_run"
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        md_content = "# Audit Report\n\n## Summary\n"
        (run_dir / "audit_report.md").write_text(md_content)
        with patch("main._OUTPUT_ROOT", tmp_path):
            resp = await client.get(f"/api/download/{run_id}/report")
        assert resp.status_code == 200
        assert b"# Audit Report" in resp.content


# ---------------------------------------------------------------------------
# POST /api/run (mocked pipeline)
# ---------------------------------------------------------------------------

class TestRunEndpoint:
    """
    The pipeline itself is tested exhaustively in other test files.
    Here we mock pipeline.stream and test only the SSE wiring in main.py.
    """

    def _mock_stream_output(self, run_id: str = "mock_run_001"):
        """Fake LangGraph stream output — list of (node_name, state_delta) tuples."""
        return [
            ("ingestion",  {"audit_log": [{"step": "ingestion", "status": "ok"}]}),
            ("profiling",  {"profile": []}),
            ("decision",   {"cleaning_plan": []}),
            ("execution",  {"current_dataset": None, "iteration": 1}),
            ("validation", {"quality_score": {"overall": 0.93, "missing_score": 0.95,
                                               "outlier_score": 0.91, "skewness_score": 0.90,
                                               "iteration": 1, "details": {}},
                            "quality_history": []}),
            ("output",     {
                "output_paths": {
                    "run_id":         run_id,
                    "csv":            f"/data/output/{run_id}/cleaned_test.csv",
                    "report":         f"/data/output/{run_id}/audit_report.md",
                    "explainability": f"/data/output/{run_id}/explainability.json",
                    "out_dir":        f"/data/output/{run_id}",
                },
                "explainability": {
                    "run_id":      run_id,
                    "shape_before": [100, 5],
                    "shape_after":  [98, 4],
                    "final_score": {"overall": 0.93},
                },
            }),
        ]

    @pytest.mark.asyncio
    async def test_run_streams_sse_events(self, client, tmp_path):
        # Upload a file first
        upload_resp = await client.post(
            "/api/upload",
            files=_upload_file(_csv_bytes()),
        )
        assert upload_resp.status_code == 200
        session_id = upload_resp.json()["session_id"]

        mock_stream = MagicMock(return_value=self._mock_stream_output())
        with patch("main.pipeline") as mock_pipeline:
            mock_pipeline.stream = mock_stream
            resp = await client.post(
                "/api/run",
                json={
                    "session_id": session_id,
                    "mode": "rules",
                    "api_key": "",
                    "max_iterations": 3,
                    "quality_threshold": 0.90,
                },
            )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    @pytest.mark.asyncio
    async def test_run_emits_done_event(self, client, tmp_path):
        upload_resp = await client.post(
            "/api/upload", files=_upload_file(_csv_bytes())
        )
        session_id = upload_resp.json()["session_id"]

        mock_stream = MagicMock(return_value=self._mock_stream_output())
        with patch("main.pipeline") as mock_pipeline:
            mock_pipeline.stream = mock_stream
            resp = await client.post(
                "/api/run",
                json={"session_id": session_id, "mode": "rules",
                      "api_key": "", "max_iterations": 3, "quality_threshold": 0.90},
            )

        text = resp.text
        assert "event: done" in text

    @pytest.mark.asyncio
    async def test_run_emits_progress_events(self, client):
        upload_resp = await client.post(
            "/api/upload", files=_upload_file(_csv_bytes())
        )
        session_id = upload_resp.json()["session_id"]

        mock_stream = MagicMock(return_value=self._mock_stream_output())
        with patch("main.pipeline") as mock_pipeline:
            mock_pipeline.stream = mock_stream
            resp = await client.post(
                "/api/run",
                json={"session_id": session_id, "mode": "rules",
                      "api_key": "", "max_iterations": 3, "quality_threshold": 0.90},
            )

        text = resp.text
        assert "event: progress" in text

    @pytest.mark.asyncio
    async def test_run_with_bad_session_emits_error(self, client):
        resp = await client.post(
            "/api/run",
            json={"session_id": "does-not-exist", "mode": "rules",
                  "api_key": "", "max_iterations": 3, "quality_threshold": 0.90},
        )
        assert resp.status_code == 200   # SSE always 200
        assert "event: error" in resp.text

    @pytest.mark.asyncio
    async def test_done_event_contains_run_id(self, client):
        upload_resp = await client.post(
            "/api/upload", files=_upload_file(_csv_bytes())
        )
        session_id = upload_resp.json()["session_id"]

        mock_stream = MagicMock(return_value=self._mock_stream_output("mock_run_001"))
        with patch("main.pipeline") as mock_pipeline:
            mock_pipeline.stream = mock_stream
            resp = await client.post(
                "/api/run",
                json={"session_id": session_id, "mode": "rules",
                      "api_key": "", "max_iterations": 3, "quality_threshold": 0.90},
            )

        # Find the "done" event data line
        done_data = None
        for line in resp.text.splitlines():
            if line.startswith("data:") and "mock_run_001" in line:
                done_data = json.loads(line[5:].strip())
                break

        assert done_data is not None
        assert done_data["run_id"] == "mock_run_001"