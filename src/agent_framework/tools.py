"""Validated tool registry, single-attempt execution, and evidence reduction."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Iterable

from pydantic import BaseModel

from .state import AgentState, Effect, Evidence, ToolCall, ToolOutput, ToolResult, fingerprint


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[[BaseModel], ToolOutput]
    effect: Effect = "read_only"
    version: str = "1"
    evidence_key: str | None = None
    invalidates: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    requires_passed: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]{0,63}", self.name):
            raise ValueError("Tool names must be identifiers with at most 64 characters")
        if not self.description or not self.version:
            raise ValueError("Tool description and version are required")
        if not isinstance(self.input_model, type) or not issubclass(self.input_model, BaseModel):
            raise ValueError("input_model must be a Pydantic BaseModel class")
        if self.input_model.model_config.get("extra") != "forbid":
            raise ValueError("Tool input models must set extra='forbid'")
        if not callable(self.handler):
            raise ValueError("handler must be callable")
        if self.effect not in ("read_only", "local_compute", "external_write"):
            raise ValueError("Unknown effect category")
        for values in (self.invalidates, self.requires, self.requires_passed):
            if not isinstance(values, tuple) or any(not isinstance(v, str) or not v for v in values):
                raise ValueError("Evidence dependencies must be tuples of nonempty keys")
        if self.evidence_key is not None and not self.evidence_key:
            raise ValueError("evidence_key must be nonempty")


@dataclass(frozen=True)
class PreparedCall:
    call: ToolCall
    spec: ToolSpec
    arguments: BaseModel
    input_fingerprint: str


def _execution_values(value):
    """Capture executed field values without lossy serializers or exclusions.

    Input models must ultimately contain finite JSON values. Private model
    state and opaque Python objects have no stable public approval contract.
    """
    if isinstance(value, BaseModel):
        if getattr(value, "__pydantic_private__", None):
            raise ValueError("Private tool input attributes cannot be approval-bound")
        fields = {name: _execution_values(getattr(value, name)) for name in type(value).model_fields}
        extras = getattr(value, "__pydantic_extra__", None) or {}
        if extras:
            fields.update({name: _execution_values(item) for name, item in extras.items()})
        return fields
    if isinstance(value, dict):
        return {key: _execution_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_execution_values(item) for item in value]
    # fingerprint performs the final finite-JSON check (including dict keys).
    return value


class ToolRegistry:
    def __init__(self, tools: Iterable[ToolSpec]) -> None:
        self._tools: dict[str, ToolSpec] = {}
        for spec in tools:
            if spec.name in self._tools:
                raise ValueError(f"Duplicate tool name: {spec.name}")
            self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(self._tools.values())

    def descriptions(self) -> list[dict]:
        return [
            {"name": s.name, "description": s.description,
             "parameters": s.input_model.model_json_schema(), "effect": s.effect, "version": s.version}
            for s in self._tools.values()
        ]

    def prepare(self, call: ToolCall, context_id: str) -> PreparedCall:
        spec = self.get(call.tool_name)
        if spec is None:
            raise ValueError("UNKNOWN_TOOL")
        # Revalidate a detached value, including previously constructed models.
        validated_call = ToolCall.model_validate(call.model_dump(mode="json", warnings=False))
        arguments = spec.input_model.model_validate(validated_call.arguments)
        canonical = _execution_values(arguments)
        digest = fingerprint({"tool": spec.name, "version": spec.version,
                              "context_id": context_id, "arguments": canonical})
        return PreparedCall(validated_call, spec, arguments, digest)

    def execute(self, prepared: PreparedCall) -> ToolResult:
        common = dict(call_id=prepared.call.call_id, tool_name=prepared.spec.name,
                      allowed=True, input_fingerprint=prepared.input_fingerprint,
                      effect=prepared.spec.effect,
                      external_action_started=prepared.spec.effect == "external_write")
        try:
            output = prepared.spec.handler(prepared.arguments.model_copy(deep=True))
            if not isinstance(output, ToolOutput):
                raise TypeError("Handlers must return ToolOutput")
            output = ToolOutput.model_validate(output.model_dump(mode="json", warnings=False))
            return ToolResult(status="success", output=output, **common)
        except Exception:
            # Exception messages may contain URLs, passwords, or request bodies.
            return ToolResult(status="failed", fail_reasons=["TOOL_EXECUTION_FAILED"], **common)


def apply_result(state: AgentState, result: ToolResult, spec: ToolSpec | None,
                 registry: ToolRegistry | None = None) -> AgentState:
    updated = state.model_copy(deep=True)
    updated.history.append(result.model_copy(deep=True))
    if not result.allowed or spec is None:
        return updated
    invalid = set(spec.invalidates)
    if spec.evidence_key:
        invalid.add(spec.evidence_key)
    if registry is not None:
        # Invalidation includes transitive dependants, even after failed attempts.
        while True:
            previous = len(invalid)
            for candidate in registry.specs():
                if candidate.evidence_key and invalid.intersection(candidate.requires + candidate.requires_passed):
                    invalid.add(candidate.evidence_key)
            if len(invalid) == previous:
                break
    for key in invalid:
        updated.evidence.pop(key, None)
    if result.status == "success" and result.output is not None and spec.evidence_key:
        updated.evidence[spec.evidence_key] = Evidence(
            key=spec.evidence_key, source_tool=spec.name,
            input_fingerprint=result.input_fingerprint,
            output=result.output.model_copy(deep=True),
        )
    return updated
