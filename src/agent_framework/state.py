"""Domain-neutral state and typed execution evidence."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

Effect = Literal["read_only", "local_compute", "external_write"]


def _check_json(value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        json.dumps(value, allow_nan=False)
        return
    if isinstance(value, list):
        for item in value:
            _check_json(item)
        return
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        for item in value.values():
            _check_json(item)
        return
    raise ValueError("Expected finite JSON values with string object keys")


def fingerprint(value: Any) -> str:
    """Hash canonical finite JSON; never hash reprs or filesystem paths implicitly."""
    _check_json(value)
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class ToolCall(StrictModel):
    call_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1, max_length=128)
    tool_name: str = Field(min_length=1, max_length=128)
    arguments: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def json_arguments(self) -> ToolCall:
        _check_json(self.arguments)
        return self


class Decision(StrictModel):
    calls: list[ToolCall] = Field(default_factory=list)
    answer: str = ""


class ToolOutput(StrictModel):
    data: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float | int | bool] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    evaluation_passed: bool | None = None

    @model_validator(mode="after")
    def json_data(self) -> ToolOutput:
        _check_json(self.data)
        _check_json(self.metrics)
        return self


class ToolResult(StrictModel):
    call_id: str
    tool_name: str
    status: Literal["success", "failed", "blocked"]
    allowed: bool
    input_fingerprint: str = ""
    effect: Effect = "read_only"
    external_action_started: bool = False
    output: ToolOutput | None = None
    fail_reasons: list[str] = Field(default_factory=list)


class Evidence(StrictModel):
    key: str
    source_tool: str
    input_fingerprint: str
    output: ToolOutput


class AgentState(StrictModel):
    task_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1)
    context_id: str = Field(min_length=1, max_length=128)
    goal: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    stage: str = "initialized"
    steps: int = Field(default=0, ge=0)
    evidence: dict[str, Evidence] = Field(default_factory=dict)
    history: list[ToolResult] = Field(default_factory=list)
    retrieved: list[dict[str, Any]] = Field(default_factory=list)
    reflections: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    final_answer: str = ""
    stop_reason: str = ""

    @model_validator(mode="after")
    def json_metadata(self) -> AgentState:
        _check_json(self.metadata)
        _check_json(self.retrieved)
        return self
