"""Explicit caller-issued approvals; user prose never grants authority."""

from __future__ import annotations

from threading import Lock
from typing import Iterable
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .state import AgentState
from .tools import PreparedCall


class ApprovalGrant(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str = Field(min_length=1)
    context_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    input_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    grant_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1)


class ExecutionPolicy:
    """One-process approval ledger, owned by trusted application code.

    A consumed grant is never restored after failure. This is not a distributed
    lock or durable exactly-once execution system. Instances should not be
    reconstructed from old grants to resume external actions.
    """

    def __init__(self, approvals: Iterable[ApprovalGrant] = ()) -> None:
        self._grants = tuple(ApprovalGrant.model_validate(g.model_dump()) for g in approvals)
        if len({g.grant_id for g in self._grants}) != len(self._grants):
            raise ValueError("Duplicate approval grant_id")
        self._consumed: set[str] = set()
        self._lock = Lock()

    def check(self, prepared: PreparedCall, state: AgentState) -> str | None:
        if any(key not in state.evidence for key in prepared.spec.requires):
            return "REQUIRED_EVIDENCE_MISSING"
        if any(key not in state.evidence or state.evidence[key].output.evaluation_passed is not True
               for key in prepared.spec.requires_passed):
            return "PASSING_EVIDENCE_REQUIRED"
        if prepared.spec.effect != "external_write":
            return None
        with self._lock:
            for grant in self._grants:
                if (grant.grant_id not in self._consumed
                        and grant.task_id == state.task_id
                        and grant.context_id == state.context_id
                        and grant.tool_name == prepared.spec.name
                        and grant.input_fingerprint == prepared.input_fingerprint):
                    self._consumed.add(grant.grant_id)
                    return None
        return "EXPLICIT_APPROVAL_REQUIRED"
