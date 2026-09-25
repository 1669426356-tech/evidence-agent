"""Demonstrate an explicit grant with a mock write; no device or network access."""

import json

from pydantic import BaseModel, ConfigDict

from agent_framework import (
    AgentRunner, ApprovalGrant, Decision, ExecutionPolicy, ScriptedModel,
    ToolCall, ToolOutput, ToolRegistry, ToolSpec,
)


class MockInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int


def main() -> None:
    observed: list[int] = []

    def mock_write(value: MockInput) -> ToolOutput:
        observed.append(value.value)
        return ToolOutput(data={"mock_only": True, "observed": value.value})

    tools = ToolRegistry([ToolSpec("mock_write", "Mock external action for teaching only", MockInput,
                                   mock_write, effect="external_write")])
    call = ToolCall(tool_name="mock_write", arguments={"value": 5})
    prepared = tools.prepare(call, "approval-example")
    # Application code constructs this only after the user explicitly confirms
    # the displayed task, tool, and exact input. Model prose cannot create it.
    grant = ApprovalGrant(task_id="example-task", context_id="approval-example", tool_name="mock_write",
                          input_fingerprint=prepared.input_fingerprint)
    state = AgentRunner(
        ScriptedModel([Decision(calls=[call]), Decision(answer="Done")]), tools,
        policy=ExecutionPolicy([grant]),
    ).run("Demonstrate a caller-issued mock approval", context_id="approval-example", task_id="example-task")
    print(json.dumps({"mode": "mock_only", "statuses": [r.status for r in state.history], "observed": observed}))


if __name__ == "__main__":
    main()
