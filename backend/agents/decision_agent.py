"""
backend/agents/decision_agent.py

Decision Agent — Phase 3
The heart of the project.  Given a list of ColumnProfiles, this agent produces
a CleaningPlan: one or more CleaningStep dicts, each with an action, params,
and a human-readable rationale.

Two modes
---------
mode = "llm"   → Ask Claude to reason a context-aware plan.  Falls back to
                  the rules engine automatically if the API call fails.
mode = "rules" → Deterministic heuristics only (fast, no API key needed).

LLM prompt design
-----------------
The prompt is deliberately structured so that Claude returns *only* a JSON
array.  We never ask for markdown fences or explanation outside the array —
this makes robust parsing trivial.  A Pydantic-style schema comment is
embedded in the prompt so the model understands the expected shape.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from pipeline.state import ColumnProfile, CleaningStep, PipelineState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action catalogue
# These are the *only* action strings the execution agent will understand.
# Keeping them here (single source of truth) avoids drift.
# ---------------------------------------------------------------------------

VALID_ACTIONS: frozenset[str] = frozenset({
    # Missing value handling
    "impute_mean",
    "impute_median",
    "impute_mode",
    "impute_constant",   # params: {"value": <fill_value>}
    "drop_column",

    # Outlier handling
    "clip_iqr",          # params: {"factor": 1.5}  — clip to whiskers
    "winsorise",         # params: {"lower": 0.01, "upper": 0.99}

    # Encoding
    "encode_onehot",
    "encode_ordinal",    # params: {"order": [...]}
    "encode_binary",     # for already-binary columns that are stored as strings

    # Scaling
    "scale_standard",    # StandardScaler
    "scale_minmax",      # MinMaxScaler
    "scale_robust",      # RobustScaler (better for skewed/outlier data)

    # Transforms
    "log_transform",     # log1p — only valid for non-negative numeric
    "sqrt_transform",    # for moderately skewed non-negative data

    # No action
    "keep",              # explicitly mark column as fine as-is
})


# ---------------------------------------------------------------------------
# Rules engine (deterministic fallback)
# ---------------------------------------------------------------------------

def _rules_decision(profile: ColumnProfile) -> CleaningStep:
    """
    Produce a single CleaningStep using pure heuristics.  The logic is
    deliberately conservative — it will never destroy data.

    Decision tree (evaluated top-to-bottom, first match wins):
    1.  missing_pct == 1.0          → drop_column
    2.  cardinality == "identifier" → keep  (ID columns; don't touch)
    3.  missing_pct > 0 AND numeric → impute_median  (robust to skew/outliers)
    4.  missing_pct > 0 AND object  → impute_mode
    5.  outlier_ratio > 0.05 AND skew > 1.0 → log_transform + clip_iqr note
    6.  outlier_ratio > 0.05        → clip_iqr
    7.  |skewness| > 1.0 AND non-negative → log_transform
    8.  cardinality == "low"/"binary" AND object → encode_onehot
    9.  numeric AND high variance   → scale_robust  (if outliers present)
       numeric AND no outliers      → scale_standard
    10. fallback                    → keep
    """
    col = profile["column"]
    dtype = profile.get("dtype", "object")
    missing = profile.get("missing_pct") or 0.0
    skew = profile.get("skewness")
    outliers = profile.get("outlier_ratio")
    cardinality = profile.get("cardinality_hint", "medium")
    ns = profile.get("numeric_summary") or {}
    is_numeric = skew is not None  # profiling only sets skewness for numeric cols

    # --- 1. Completely null column ------------------------------------------
    if missing == 1.0:
        return CleaningStep(
            column=col,
            action="drop_column",
            rationale="Column is 100 % null — no information to preserve.",
            params={},
        )

    # --- 2. Identifier column -----------------------------------------------
    if cardinality == "identifier":
        return CleaningStep(
            column=col,
            action="keep",
            rationale=(
                "Column appears to be a unique identifier (unique_count ≈ row count). "
                "Identifiers should not be scaled or encoded."
            ),
            params={},
        )

    # --- 3 & 4. Missing value imputation ------------------------------------
    if missing > 0:
        if is_numeric:
            return CleaningStep(
                column=col,
                action="impute_median",
                rationale=(
                    f"{missing:.1%} of values are missing.  Median imputation chosen "
                    "because it is robust to the skewness and outliers often present "
                    "in real-world numeric data."
                ),
                params={},
            )
        else:
            return CleaningStep(
                column=col,
                action="impute_mode",
                rationale=(
                    f"{missing:.1%} of values are missing in this categorical column.  "
                    "Mode imputation preserves the most common category."
                ),
                params={},
            )

    # --- 5. High outlier ratio + high skew ---------------------------------
    if is_numeric and outliers is not None and outliers > 0.05 and skew is not None and abs(skew) > 1.0:
        # Check if safe to log-transform (min must be >= 0)
        col_min = ns.get("min")
        if col_min is not None and col_min >= 0:
            return CleaningStep(
                column=col,
                action="log_transform",
                rationale=(
                    f"Column has {outliers:.1%} outliers and skewness of {skew:.2f}.  "
                    "log1p transform reduces right-skew and compresses extreme values, "
                    "addressing both issues simultaneously."
                ),
                params={},
            )
        else:
            return CleaningStep(
                column=col,
                action="clip_iqr",
                rationale=(
                    f"Column has {outliers:.1%} outliers and skewness of {skew:.2f} "
                    "but contains negative values, so log transform is not safe.  "
                    "IQR clipping used instead."
                ),
                params={"factor": 1.5},
            )

    # --- 6. Outliers only ---------------------------------------------------
    if is_numeric and outliers is not None and outliers > 0.05:
        return CleaningStep(
            column=col,
            action="clip_iqr",
            rationale=(
                f"{outliers:.1%} of values fall outside 1.5×IQR bounds.  "
                "IQR clipping is non-destructive and preserves the distribution shape."
            ),
            params={"factor": 1.5},
        )

    # --- 7. Skew only -------------------------------------------------------
    if is_numeric and skew is not None and abs(skew) > 1.0:
        col_min = ns.get("min")
        if col_min is not None and col_min >= 0:
            return CleaningStep(
                column=col,
                action="log_transform",
                rationale=(
                    f"Skewness of {skew:.2f} indicates a right-skewed distribution.  "
                    "log1p transform normalises the distribution for downstream ML models."
                ),
                params={},
            )

    # --- 8. Categorical encoding --------------------------------------------
    if not is_numeric and cardinality in ("binary", "low"):
        return CleaningStep(
            column=col,
            action="encode_onehot",
            rationale=(
                f"Categorical column with {cardinality} cardinality.  "
                "One-hot encoding is appropriate for nominal variables with few levels."
            ),
            params={},
        )

    # --- 9. Scaling ---------------------------------------------------------
    if is_numeric:
        if outliers is not None and outliers > 0.01:
            return CleaningStep(
                column=col,
                action="scale_robust",
                rationale=(
                    "Numeric column with some outliers present.  "
                    "RobustScaler (based on median/IQR) is less sensitive to extremes "
                    "than StandardScaler."
                ),
                params={},
            )
        return CleaningStep(
            column=col,
            action="scale_standard",
            rationale=(
                "Numeric column with no significant outliers or skew.  "
                "StandardScaler centres and normalises the distribution."
            ),
            params={},
        )

    # --- 10. Fallback -------------------------------------------------------
    return CleaningStep(
        column=col,
        action="keep",
        rationale=(
            "No cleaning action required based on profile statistics."
        ),
        params={},
    )


def build_rules_plan(profiles: list[ColumnProfile]) -> list[CleaningStep]:
    """Apply _rules_decision to every column profile."""
    return [_rules_decision(p) for p in profiles]


# ---------------------------------------------------------------------------
# LLM prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a senior data scientist specialising in automated data cleaning for \
machine learning pipelines.

You will receive a JSON array of column profiles from a dataset.  Your task is \
to produce a cleaning plan: one CleaningStep per column.

IMPORTANT — respond with ONLY a valid JSON array.  No markdown, no prose, no \
code fences.  The array must be parseable by json.loads().

Each element must match this schema exactly:
{
  "column":    "<column name>",
  "action":    "<one of the valid actions below>",
  "rationale": "<1–3 sentence explanation of your reasoning for a non-technical audience>",
  "params":    {}   // or a params object where the action requires one
}

Valid actions (use exactly these strings):
  impute_mean, impute_median, impute_mode, impute_constant,
  drop_column, clip_iqr, winsorise, encode_onehot, encode_ordinal,
  encode_binary, scale_standard, scale_minmax, scale_robust,
  log_transform, sqrt_transform, keep

Params required per action:
  impute_constant  → {"value": <fill value>}
  clip_iqr         → {"factor": 1.5}   (default; adjust if justified)
  winsorise        → {"lower": 0.01, "upper": 0.99}
  encode_ordinal   → {"order": [...]}  (list of category values, low → high)
  (all others)     → {}

Decision guidelines:
- Prefer median imputation over mean for skewed or outlier-heavy columns.
- Use log_transform only when the column minimum is ≥ 0.
- Use scale_robust when outlier_ratio > 0.01; scale_standard otherwise.
- Never apply encoding to numeric columns.
- Mark identifier columns (cardinality_hint = "identifier") as "keep".
- Drop columns that are 100 % null.
- Give a rationale that explains *why* this action was chosen for *this column*, \
  not just what the action does.
- If this is iteration > 1, the profile reflects data from the *previous* cleaning \
  pass.  Focus on remaining issues only.
"""


def _build_user_prompt(
    profiles: list[ColumnProfile],
    iteration: int,
    previous_plan: list[CleaningStep] | None,
) -> str:
    # Serialise profiles to JSON (drop any non-serialisable fields gracefully)
    safe_profiles = []
    for p in profiles:
        safe = {k: v for k, v in p.items() if k != "profiling_error"}
        safe_profiles.append(safe)

    prompt_parts = [
        f"Iteration: {iteration}",
        "",
        "Column profiles:",
        json.dumps(safe_profiles, indent=2, default=str),
    ]

    if previous_plan and iteration > 1:
        prompt_parts += [
            "",
            "Previous cleaning plan (iteration already executed):",
            json.dumps(previous_plan, indent=2, default=str),
            "",
            "The quality score did not reach the threshold.  "
            "Revise your plan to address remaining data quality issues.",
        ]

    return "\n".join(prompt_parts)


# ---------------------------------------------------------------------------
# LLM call (langchain-google-genai / Gemini)
# ---------------------------------------------------------------------------

def _call_llm(
    api_key: str,
    profiles: list[ColumnProfile],
    iteration: int,
    previous_plan: list[CleaningStep] | None,
) -> list[CleaningStep]:
    """
    Call Gemini via langchain-google-genai and parse the response.
    Raises on failure so the caller can fall back to rules.
    """
    # Lazy import — langchain-google-genai is only needed in LLM mode
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError as exc:
        raise RuntimeError(
            "langchain-google-genai is not installed.  "
            "Run: pip install langchain-google-genai"
        ) from exc

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=api_key,
        temperature=0,          # deterministic — we want consistent plans
        max_output_tokens=4096,
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_build_user_prompt(profiles, iteration, previous_plan)),
    ]

    response = llm.invoke(messages)
    raw_text: str = response.content

    # Strip accidental markdown fences if the model adds them anyway
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())

    try:
        plan_raw: list[dict] = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned non-JSON output.  Raw response:\n{raw_text}"
        ) from exc

    return _validate_and_coerce_plan(plan_raw, profiles)


# ---------------------------------------------------------------------------
# Plan validation & normalisation
# ---------------------------------------------------------------------------

def _validate_and_coerce_plan(
    raw: list[dict],
    profiles: list[ColumnProfile],
) -> list[CleaningStep]:
    """
    Ensure every element of the raw plan has valid fields.
    - Unknown actions → replaced with "keep" (safe default)
    - Missing columns → filled in with a "keep" step
    - Extra columns (not in profiles) → silently dropped
    """
    profiled_cols = {p["column"] for p in profiles}
    plan_cols: set[str] = set()
    cleaned: list[CleaningStep] = []

    for item in raw:
        col = item.get("column", "")
        if col not in profiled_cols:
            logger.warning("LLM returned step for unknown column '%s' — skipped.", col)
            continue

        action = item.get("action", "keep")
        if action not in VALID_ACTIONS:
            logger.warning(
                "LLM returned unknown action '%s' for '%s' — defaulting to 'keep'.",
                action, col,
            )
            action = "keep"

        cleaned.append(CleaningStep(
            column=col,
            action=action,
            rationale=item.get("rationale", "No rationale provided."),
            params=item.get("params") or {},
        ))
        plan_cols.add(col)

    # Fill in any columns the LLM missed
    for col in profiled_cols - plan_cols:
        logger.warning("LLM omitted column '%s' — inserting 'keep' step.", col)
        cleaned.append(CleaningStep(
            column=col,
            action="keep",
            rationale="Column was not addressed by the LLM; defaulting to no action.",
            params={},
        ))

    return cleaned


# ---------------------------------------------------------------------------
# Public node function
# ---------------------------------------------------------------------------

def run_decision(state: PipelineState) -> dict[str, Any]:
    """
    LangGraph node: decision.

    Reads  : state["profile"], state["mode"], state["api_key"],
             state["iteration"], state["cleaning_plan"] (previous, if any)
    Writes : state["cleaning_plan"], state["audit_log"]
    """
    audit_log: list[dict] = list(state.get("audit_log", []))
    entry: dict[str, Any] = {
        "step": "decision",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "iteration": state.get("iteration", 0),
    }

    try:
        profiles: list[ColumnProfile] = state.get("profile") or []
        if not profiles:
            raise ValueError("profile is empty — profiling must run before decision.")

        mode: str = state.get("mode", "rules")
        iteration: int = state.get("iteration", 0)
        previous_plan: list[CleaningStep] | None = state.get("cleaning_plan")

        plan: list[CleaningStep]
        plan_source: str

        if mode == "llm":
            api_key = state.get("api_key", "")
            if not api_key:
                logger.warning("mode=llm but no google_api_key — falling back to rules.")
                plan = build_rules_plan(profiles)
                plan_source = "rules_fallback_no_key"
            else:
                try:
                    logger.info("Calling LLM for cleaning plan (iteration %d)…", iteration)
                    plan = _call_llm(api_key, profiles, iteration, previous_plan)
                    plan_source = "llm"
                    logger.info("LLM plan received (%d steps).", len(plan))
                except Exception as llm_exc:  # noqa: BLE001
                    logger.warning(
                        "LLM call failed (%s) — falling back to rules engine.", llm_exc
                    )
                    plan = build_rules_plan(profiles)
                    plan_source = "rules_fallback_llm_error"
                    entry["llm_error"] = str(llm_exc)
        else:
            plan = build_rules_plan(profiles)
            plan_source = "rules"

        # Summarise the plan for the audit log
        action_summary: dict[str, list[str]] = {}
        for step in plan:
            action_summary.setdefault(step["action"], []).append(step["column"])

        entry.update({
            "status": "ok",
            "plan_source": plan_source,
            "n_steps": len(plan),
            "action_summary": action_summary,
        })
        logger.info(
            "Decision OK — %d steps via %s | actions: %s",
            len(plan),
            plan_source,
            {k: len(v) for k, v in action_summary.items()},
        )

    except Exception as exc:  # noqa: BLE001
        entry.update({"status": "error", "error": str(exc)})
        logger.exception("Decision agent failed.")
        audit_log.append(entry)
        raise

    audit_log.append(entry)
    return {
        "cleaning_plan": plan,
        "audit_log": audit_log,
    }