"""A deterministic local walkthrough, not a benchmark of an LLM."""

from __future__ import annotations

import argparse
import json

from pydantic import BaseModel, ConfigDict, Field

from .graph import AgentRunner
from .memory import InMemoryExperienceStore
from .models import ScriptedModel
from .runtime import save_state
from .state import Decision, ToolCall, ToolOutput
from .tools import ToolRegistry, ToolSpec


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    x: float = Field(ge=-10, le=10)


def evaluate(candidate: Candidate) -> ToolOutput:
    objective = (candidate.x - 3.0) ** 2
    return ToolOutput(data={"x": candidate.x}, metrics={"objective": objective},
                      evaluation_passed=objective <= 0.01)


def run_demo():
    tools = ToolRegistry([ToolSpec(
        name="evaluate_candidate", description="Evaluate (x - 3)^2 for a bounded input.",
        input_model=Candidate, handler=evaluate, effect="local_compute", evidence_key="candidate",
    )])
    model = ScriptedModel([
        Decision(calls=[ToolCall(call_id="candidate-0", tool_name="evaluate_candidate", arguments={"x": 0.0})]),
        Decision(calls=[ToolCall(call_id="candidate-3", tool_name="evaluate_candidate", arguments={"x": 3.0})]),
        Decision(answer="Scripted example complete. Inspect tool evidence for the evaluation result."),
    ])
    with InMemoryExperienceStore() as memory:
        state = AgentRunner(model, tools, experience_store=memory, max_steps=4).run(
            "Find a candidate with objective <= 0.01", context_id="local-example",
        )
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Optional JSON state snapshot path (no automatic resume)")
    args = parser.parse_args()
    state = run_demo()
    if args.output:
        save_state(state, args.output)
    print(json.dumps({
        "mode": "scripted_offline", "stop_reason": state.stop_reason, "steps": state.steps,
        "calls": len(state.history), "reflections": len(state.reflections),
        "evaluation_passed": state.evidence["candidate"].output.evaluation_passed,
        "objective": state.evidence["candidate"].output.metrics["objective"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
