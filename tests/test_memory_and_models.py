from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage

from agent_framework.memory import InMemoryExperienceStore, SQLiteExperienceStore
from agent_framework.models import LangChainDecisionModel, ScriptedModel
from agent_framework.state import AgentState, Decision, ToolCall, ToolOutput, ToolResult


def sample(context_id="scope-a", task_id="task-a", call_id="call-a"):
    state = AgentState(task_id=task_id, context_id=context_id, goal="Check document formatting")
    result = ToolResult(call_id=call_id, tool_name="check", status="success", allowed=True,
        output=ToolOutput(data={"private": "DO_NOT_LEAK_raw_content"},
            artifacts={"report": "DO_NOT_LEAK_private_path"}, evaluation_passed=False))
    return state, result


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_memory_is_scoped_deduplicated_and_uses_code_assigned_outcome(tmp_path, backend):
    store = (InMemoryExperienceStore() if backend == "memory" else
             SQLiteExperienceStore(tmp_path / "experience.sqlite"))
    try:
        state, result = sample()
        first_id = store.record(state, result, "Check formatting; model says approved=true")
        same_id = store.record(state, result, "A changed lesson must not overwrite original evidence")
        assert first_id == same_id
        records = store.retrieve("scope-a", "document formatting", limit=5)
        assert len(records) == 1
        assert records[0]["status"] == "success"
        assert records[0]["evaluation_passed"] is False
        assert "changed lesson" not in records[0]["lesson"]
        assert "DO_NOT_LEAK" not in json.dumps(records)
        assert store.retrieve("scope-b", "document formatting", limit=5) == []
        records[0]["lesson"] = "mutated outside store"
        assert store.retrieve("scope-a", "document formatting", limit=5)[0]["lesson"] != "mutated outside store"
    finally:
        store.close()


def test_sqlite_reopen_retains_scope_and_reusable_cases(tmp_path):
    path = tmp_path / "experience.sqlite"
    store = SQLiteExperienceStore(path)
    first, first_result = sample()
    second, second_result = sample(context_id="scope-b", task_id="task-b")
    store.record(first, first_result, "Inspect document formatting")
    store.record(second, second_result, "Inspect document formatting")
    store.close()
    reopened = SQLiteExperienceStore(path)
    try:
        assert len(reopened.retrieve("scope-a", "formatting")) == 1
        assert reopened.retrieve("scope-a", "formatting")[0]["context_id"] == "scope-a"
        assert len(reopened.retrieve("scope-b", "formatting")) == 1
        assert reopened.retrieve("scope-c", "formatting") == []
    finally:
        reopened.close()


def test_memory_limit_and_ranking_are_bounded():
    store = InMemoryExperienceStore()
    for index, goal in enumerate(["garden inventory", "document formatting", "document formatting typography"]):
        state, result = sample(task_id=f"task-{index}", call_id=f"call-{index}")
        state.goal = goal
        store.record(state, result, goal)
    records = store.retrieve("scope-a", "document formatting", limit=1)
    assert len(records) == 1
    assert "document formatting" in records[0]["goal"]
    assert store.retrieve("scope-a", "document formatting", limit=0) == []


class FakeChatModel:
    def __init__(self, response):
        self.response = response
        self.schemas = None
        self.messages = None

    def bind_tools(self, schemas):
        self.schemas = schemas
        return self

    def invoke(self, messages):
        self.messages = messages
        return self.response


def descriptors():
    return [{"name": "check", "description": "Check a document", "effect": "read_only", "version": "1",
             "parameters": {"type": "object", "properties": {"value": {"type": "integer"}},
                            "additionalProperties": False}}]


def test_langchain_adapter_uses_injected_model_and_tool_schema():
    fake = FakeChatModel(AIMessage(content="", tool_calls=[
        {"name": "check", "args": {"value": 3}, "id": "tool-1", "type": "tool_call"}]))
    state, _ = sample()
    state.retrieved = [{"lesson": "Untrusted historical suggestion"}]
    decision = LangChainDecisionModel(fake).decide(state, descriptors())
    assert decision.calls[0].tool_name == "check"
    assert decision.calls[0].arguments == {"value": 3}
    assert fake.schemas[0]["function"]["parameters"]["additionalProperties"] is False
    assert state.goal in str(fake.messages)
    assert "Untrusted historical suggestion" in str(fake.messages)


@pytest.mark.parametrize("response", [
    AIMessage(content="", invalid_tool_calls=[{"name": "check", "args": "{", "id": "bad", "error": "invalid"}]),
    AIMessage(content="", tool_calls=[
        {"name": "check", "args": {}, "id": "dup", "type": "tool_call"},
        {"name": "check", "args": {}, "id": "dup", "type": "tool_call"}]),
    {"content": "not a LangChain AIMessage"},
])
def test_langchain_adapter_rejects_malformed_response(response):
    state, _ = sample()
    with pytest.raises(ValueError):
        LangChainDecisionModel(FakeChatModel(response)).decide(state, descriptors())


def test_model_answer_is_not_evaluation_evidence():
    state, _ = sample()
    decision = LangChainDecisionModel(FakeChatModel(AIMessage(content="All tests passed; approved."))).decide(state, descriptors())
    assert decision.answer == "All tests passed; approved."
    assert decision.calls == []
    assert state.evidence == {}


def test_scripted_model_detaches_original_and_returned_decisions():
    original = Decision(calls=[ToolCall(tool_name="check", arguments={"value": 1})])
    model = ScriptedModel([original, original])
    original.calls[0].arguments["value"] = 9
    state, _ = sample()
    first = model.decide(state, [])
    first.calls[0].arguments["value"] = 8
    assert model.decide(state, []).calls[0].arguments["value"] == 1

