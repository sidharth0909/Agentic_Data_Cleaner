"""
backend/pipeline/graph.py

LangGraph pipeline graph — fully wired through Phase 4.

Graph topology
--------------
ingestion → profiling → decision → execution → validation
                            ↑________________________↓  (verdict == "retry")
                                        ↓  (verdict == "done")
                                     output → END

The conditional edge reads the verdict written by run_validation() into the
audit log.  It never touches quality_score directly so the logic is easy to
test in isolation.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from agents.ingestion import run_ingestion
from agents.profiling_agent import run_profiling
from agents.decision_agent import run_decision
from agents.execution_agent import run_execution
from agents.validation_agent import run_validation
from agents.output import run_output          # Phase 5 — stub until then
from pipeline.state import PipelineState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------

def should_continue(state: PipelineState) -> str:
    """
    Routing function called after validation.

    Returns
    -------
    "retry"  → loop back to decision for another cleaning pass
    "done"   → proceed to output node then END
    """
    score = state.get("quality_score")
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)
    threshold = state.get("quality_threshold", 0.90)

    if score is None:
        logger.warning("quality_score is None in should_continue — defaulting to done.")
        return "done"

    overall = score.get("overall", 0.0)

    if overall >= threshold:
        logger.info(
            "Quality threshold reached (%.4f >= %.4f) after %d iteration(s) — done.",
            overall, threshold, iteration,
        )
        return "done"

    if iteration >= max_iterations:
        logger.info(
            "Max iterations (%d) reached with score %.4f — stopping.",
            max_iterations, overall,
        )
        return "done"

    logger.info(
        "Score %.4f < threshold %.4f, iteration %d/%d — retrying.",
        overall, threshold, iteration, max_iterations,
    )
    return "retry"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Construct and compile the full pipeline graph."""
    graph = StateGraph(PipelineState)

    # --- Nodes --------------------------------------------------------------
    graph.add_node("ingestion",  run_ingestion)
    graph.add_node("profiling",  run_profiling)
    graph.add_node("decision",   run_decision)
    graph.add_node("execution",  run_execution)
    graph.add_node("validation", run_validation)
    graph.add_node("output",     run_output)

    # --- Linear edges -------------------------------------------------------
    graph.set_entry_point("ingestion")
    graph.add_edge("ingestion",  "profiling")
    graph.add_edge("profiling",  "decision")
    graph.add_edge("decision",   "execution")
    graph.add_edge("execution",  "validation")

    # --- Conditional edge (the loop) ----------------------------------------
    graph.add_conditional_edges(
        "validation",
        should_continue,
        {
            "retry": "decision",   # re-profile is skipped intentionally:
                                   # the decision agent re-reads current_dataset
                                   # indirectly via the profile in state.
                                   # Phase 5 may add a re-profile step here.
            "done":  "output",
        },
    )

    graph.add_edge("output", END)

    return graph.compile()


# Singleton — import this in main.py
pipeline = build_graph()