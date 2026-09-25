"""A bounded, dependency-injected LangGraph workflow.

Each call passes through the policy and observation nodes separately. This is
intentional: an earlier call can consume an approval or invalidate evidence
needed by the next call in the same model response.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from .memory import MAX_LESSON_LENGTH, MAX_RETRIEVAL_LIMIT, ExperienceStore, default_lesson
from .policy import ExecutionPolicy
from .state import AgentState, Decision, ToolCall, ToolResult
from .tools import PreparedCall, ToolRegistry, apply_result


class DecisionModel(Protocol):
    """Model adapter contract; a model proposes actions, never permissions."""

    def decide(self, state: AgentState, tools: list[dict]) -> Decision: ...


class Reflector(Protocol):
    """An optional lesson generator with no authority over result status."""

    def reflect(self, state: AgentState, result: ToolResult) -> str: ...


class _WorkflowState(TypedDict):
    agent: AgentState
    pending: list[ToolCall]
    seen_call_ids: set[str]
    prepared: PreparedCall | None
    result: ToolResult | None


def _detached_prepared(prepared: PreparedCall) -> PreparedCall:
    """Copy mutable call data while preserving the registered handler itself."""

    return replace(
        prepared,
        call=prepared.call.model_copy(deep=True),
        arguments=prepared.arguments.model_copy(deep=True),
    )


class AgentRunner:
    """Run fresh tasks with explicit gates and a finite decision budget.

    ``max_steps`` limits model decisions. A decision can contain at most eight
    calls, which are processed in order. An oversized decision executes no
    calls. Calls are never retried by this runner, including failed writes.

    The caller owns the experience store and its lifetime. Passing no store
    disables retrieval and recording. The policy instance is kept between
    runs so consumed grants remain consumed; this is not a durable, distributed
    approval ledger or a mechanism for resuming external actions.
    """

    MAX_CALLS_PER_DECISION = 8

    def __init__(
        self,
        model: DecisionModel,
        tools: ToolRegistry,
        *,
        policy: ExecutionPolicy | None = None,
        experience_store: ExperienceStore | None = None,
        reflector: Reflector | None = None,
        max_steps: int = 12,
        retrieval_limit: int = 5,
    ) -> None:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if (
            isinstance(retrieval_limit, bool)
            or not isinstance(retrieval_limit, int)
            or not 0 <= retrieval_limit <= MAX_RETRIEVAL_LIMIT
        ):
            raise ValueError(f"retrieval_limit must be an integer from 0 to {MAX_RETRIEVAL_LIMIT}")
        self.model = model
        self.tools = tools
        self.policy = policy if policy is not None else ExecutionPolicy()
        self.experience_store = experience_store
        self.reflector = reflector
        self.max_steps = max_steps
        self.retrieval_limit = retrieval_limit
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(_WorkflowState)
        workflow.add_node("ingest", self._ingest)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("decide", self._decide)
        workflow.add_node("gate", self._gate)
        workflow.add_node("execute", self._execute)
        workflow.add_node("observe", self._observe)
        workflow.add_node("reflect", self._reflect)
        workflow.add_edge(START, "ingest")
        workflow.add_edge("ingest", "retrieve")
        workflow.add_edge("retrieve", "decide")
        workflow.add_conditional_edges(
            "decide", self._after_decide, {"gate": "gate", END: END}
        )
        workflow.add_edge("gate", "execute")
        workflow.add_edge("execute", "observe")
        workflow.add_edge("observe", "reflect")
        workflow.add_conditional_edges(
            "reflect",
            self._after_reflect,
            {"gate": "gate", "retrieve": "retrieve", END: END},
        )
        return workflow.compile()

    def run(
        self,
        goal: str,
        *,
        context_id: str = "default",
        task_id: str | None = None,
        metadata: dict | None = None,
    ) -> AgentState:
        """Create and finish a new task; no approvals are inferred from text."""

        initial = {
            "goal": goal,
            "context_id": context_id,
            "metadata": deepcopy(metadata) if metadata is not None else {},
        }
        if task_id is not None:
            initial["task_id"] = task_id
        agent = AgentState.model_validate(initial)
        # In the worst case each decision uses retrieve + decide and eight
        # gate/execute/observe/reflect groups. Leave room for ingest and exit.
        recursion_limit = self.max_steps * (2 + 4 * self.MAX_CALLS_PER_DECISION) + 8
        result = self.graph.invoke(
            {
                "agent": agent,
                "pending": [],
                "seen_call_ids": set(),
                "prepared": None,
                "result": None,
            },
            config={"recursion_limit": recursion_limit},
        )
        return result["agent"].model_copy(deep=True)

    @staticmethod
    def _finish(agent: AgentState, reason: str) -> AgentState:
        agent.stage = "finished"
        agent.stop_reason = reason
        return agent

    @staticmethod
    def _ingest(work: _WorkflowState) -> dict:
        agent = work["agent"].model_copy(deep=True)
        agent.stage = "ingest"
        return {"agent": agent}

    def _retrieve(self, work: _WorkflowState) -> dict:
        agent = work["agent"].model_copy(deep=True)
        agent.stage = "retrieve"
        agent.retrieved = []
        if self.experience_store is not None and self.retrieval_limit:
            try:
                retrieved = self.experience_store.retrieve(
                    agent.context_id, agent.goal, limit=self.retrieval_limit
                )
                if not isinstance(retrieved, list) or any(
                    not isinstance(item, dict) for item in retrieved
                ):
                    raise ValueError("Invalid retrieval response")
                # Validate before assignment: a failing Pydantic after-validator
                # can otherwise leave its invalid assigned value on the model.
                candidate = agent.model_dump(mode="python")
                candidate["retrieved"] = deepcopy(retrieved[: self.retrieval_limit])
                agent = AgentState.model_validate(candidate)
            except Exception:
                agent.warnings.append("Experience retrieval failed; continuing without retrieved lessons.")
        return {"agent": agent}

    def _decide(self, work: _WorkflowState) -> dict:
        agent = work["agent"].model_copy(deep=True)
        agent.stage = "decide"
        if agent.steps >= self.max_steps:
            return {"agent": self._finish(agent, "max_steps"), "pending": []}
        agent.steps += 1
        try:
            proposed = self.model.decide(
                agent.model_copy(deep=True), deepcopy(self.tools.descriptions())
            )
        except Exception:
            agent.warnings.append("Model decision failed; task stopped.")
            return {"agent": self._finish(agent, "model_error"), "pending": []}
        try:
            raw_calls = (
                proposed.calls
                if isinstance(proposed, Decision)
                else proposed.get("calls", []) if isinstance(proposed, dict) else None
            )
            if isinstance(raw_calls, (list, tuple)) and len(raw_calls) > self.MAX_CALLS_PER_DECISION:
                agent.warnings.append("Model decision exceeded the tool call limit; no calls were executed.")
                return {
                    "agent": self._finish(agent, "too_many_tool_calls"),
                    "pending": [],
                }
            # Revalidate a serialized copy even when the adapter returns a
            # Decision instance whose nested fields it may have mutated.
            decision = Decision.model_validate(
                proposed.model_dump(mode="python", warnings=False)
                if isinstance(proposed, Decision)
                else proposed
            )
            if len(decision.calls) > self.MAX_CALLS_PER_DECISION:
                agent.warnings.append("Model decision exceeded the tool call limit; no calls were executed.")
                return {
                    "agent": self._finish(agent, "too_many_tool_calls"),
                    "pending": [],
                }
        except Exception:
            agent.warnings.append("Model returned an invalid decision; task stopped.")
            return {"agent": self._finish(agent, "invalid_decision"), "pending": []}
        if not decision.calls:
            agent.final_answer = decision.answer
            return {"agent": self._finish(agent, "completed"), "pending": []}
        return {"agent": agent, "pending": deepcopy(decision.calls)}

    @staticmethod
    def _after_decide(work: _WorkflowState) -> str:
        return END if work["agent"].stop_reason else "gate"

    def _gate(self, work: _WorkflowState) -> dict:
        agent = work["agent"].model_copy(deep=True)
        agent.stage = "gate"
        call = work["pending"][0].model_copy(deep=True)
        remaining = work["pending"][1:]
        seen = set(work["seen_call_ids"])
        prepared = None
        blocked_reason = None
        if call.call_id in seen:
            blocked_reason = "Duplicate tool call identifier; call was not executed."
        else:
            # Invalid and denied calls also reserve their IDs within this run.
            seen.add(call.call_id)
            try:
                prepared = self.tools.prepare(call, agent.context_id)
            except Exception:
                blocked_reason = "Tool call is unknown or its arguments are invalid."
            if prepared is not None:
                try:
                    blocked_reason = self.policy.check(
                        _detached_prepared(prepared), agent.model_copy(deep=True)
                    )
                    if blocked_reason is not None and (
                        not isinstance(blocked_reason, str) or not blocked_reason.strip()
                    ):
                        raise ValueError("Invalid policy response")
                except Exception:
                    blocked_reason = "Execution policy failed; call was blocked."
                    agent.warnings.append("Execution policy check failed; no action was authorized.")
        result = None
        if blocked_reason is not None:
            spec = prepared.spec if prepared is not None else self.tools.get(call.tool_name)
            result = ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                status="blocked",
                allowed=False,
                input_fingerprint=prepared.input_fingerprint if prepared else "",
                effect=spec.effect if spec else "read_only",
                fail_reasons=[blocked_reason],
            )
        return {
            "agent": agent,
            "pending": remaining,
            "seen_call_ids": seen,
            "prepared": prepared,
            "result": result,
        }

    def _execute(self, work: _WorkflowState) -> dict:
        agent = work["agent"].model_copy(deep=True)
        agent.stage = "execute"
        if work["result"] is not None:
            return {"agent": agent}
        prepared = work["prepared"]
        if prepared is None:
            raise RuntimeError("Workflow invariant: missing prepared call")
        try:
            result = self.tools.execute(_detached_prepared(prepared))
        except Exception:
            # Registry implementations normally wrap tool errors themselves.
            # If an extension raises, retain conservative write-attempt status.
            result = ToolResult(
                call_id=prepared.call.call_id,
                tool_name=prepared.call.tool_name,
                status="failed",
                allowed=True,
                input_fingerprint=prepared.input_fingerprint,
                effect=prepared.spec.effect,
                external_action_started=prepared.spec.effect == "external_write",
                fail_reasons=["Tool execution failed; the call will not be retried."],
            )
        return {"agent": agent, "result": result}

    def _observe(self, work: _WorkflowState) -> dict:
        agent = work["agent"].model_copy(deep=True)
        agent.stage = "observe"
        result = work["result"]
        if result is None:
            raise RuntimeError("Workflow invariant: missing tool result")
        spec = self.tools.get(result.tool_name)
        agent = apply_result(agent, result.model_copy(deep=True), spec, registry=self.tools)
        return {"agent": agent}

    def _reflect(self, work: _WorkflowState) -> dict:
        agent = work["agent"].model_copy(deep=True)
        agent.stage = "reflect"
        result = work["result"]
        if result is None:
            raise RuntimeError("Workflow invariant: missing reflection result")
        lesson = default_lesson(result)
        if self.reflector is not None:
            try:
                proposed = self.reflector.reflect(
                    agent.model_copy(deep=True), result.model_copy(deep=True)
                )
                if not isinstance(proposed, str) or not proposed.strip():
                    raise ValueError("Invalid lesson")
                candidate = proposed.strip()
                if len(candidate) > MAX_LESSON_LENGTH:
                    raise ValueError("Lesson exceeds the supported length")
                lesson = candidate
            except Exception:
                agent.warnings.append("Reflection failed; using a deterministic lesson.")
        agent.reflections.append(lesson)
        if self.experience_store is not None:
            try:
                self.experience_store.record(
                    agent.model_copy(deep=True), result.model_copy(deep=True), lesson
                )
            except Exception:
                agent.warnings.append("Experience recording failed; task state retains the result.")
        if not work["pending"] and agent.steps >= self.max_steps:
            self._finish(agent, "max_steps")
        return {"agent": agent, "prepared": None, "result": None}

    @staticmethod
    def _after_reflect(work: _WorkflowState) -> str:
        if work["agent"].stop_reason:
            return END
        return "gate" if work["pending"] else "retrieve"
