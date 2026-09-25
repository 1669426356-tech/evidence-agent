"""Small, context-scoped experience stores with code-owned provenance.

Only a bounded text lesson is supplied by a reflector. Tool output bodies,
artifact paths, and exception details are intentionally not stored here.
"""

from __future__ import annotations

import json
import re
import sqlite3
from copy import deepcopy
from os import PathLike
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .state import AgentState, ToolResult, fingerprint


MAX_RETRIEVAL_LIMIT = 100
MAX_LESSON_LENGTH = 4096


class ExperienceStore(Protocol):
    """A caller-owned store; a runner never closes an injected store."""

    def record(self, state: AgentState, result: ToolResult, lesson: str) -> str: ...

    def retrieve(self, context_id: str, goal: str, limit: int = 5) -> list[dict]: ...

    def close(self) -> None: ...


class _Experience(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    experience_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    context_id: str = Field(min_length=1)
    call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    input_fingerprint: str
    status: Literal["success", "failed", "blocked"]
    evaluation_passed: bool | None
    goal: str = Field(min_length=1)
    lesson: str = Field(min_length=1, max_length=MAX_LESSON_LENGTH)


def default_lesson(result: ToolResult) -> str:
    """Describe the recorded outcome without copying arbitrary failure text."""
    if not isinstance(result, ToolResult):
        raise TypeError("result must be a ToolResult")
    if result.status == "blocked":
        return "Execution was blocked. Satisfy the declared prerequisites before proposing another call."
    if result.status == "failed":
        if result.external_action_started:
            return "An external action may have started before failure. Verify its outcome before considering another action."
        return "Execution failed. Review the tool inputs and implementation before proposing another call."
    passed = result.output.evaluation_passed if result.output is not None else None
    if passed is True:
        return "Execution succeeded and the tool's explicit evaluation passed."
    if passed is False:
        return "Execution succeeded but the tool's explicit evaluation did not pass."
    return "Execution succeeded without an explicit evaluation result."


def _make_experience(state: AgentState, result: ToolResult, lesson: str) -> _Experience:
    if not isinstance(state, AgentState) or not isinstance(result, ToolResult):
        raise TypeError("record requires AgentState and ToolResult instances")
    # Validate again because Pydantic model_copy(update=...) does not validate.
    state = AgentState.model_validate(state.model_dump(mode="json"))
    result = ToolResult.model_validate(result.model_dump(mode="json"))
    if not isinstance(lesson, str):
        raise TypeError("lesson must be text")
    lesson = lesson.strip() or default_lesson(result)
    if len(lesson) > MAX_LESSON_LENGTH:
        raise ValueError("lesson exceeds the supported length")
    identity = {
        "task_id": state.task_id,
        "context_id": state.context_id,
        "call_id": result.call_id,
    }
    return _Experience(
        experience_id=fingerprint(identity),
        **identity,
        tool_name=result.tool_name,
        input_fingerprint=result.input_fingerprint,
        status=result.status,
        evaluation_passed=result.output.evaluation_passed if result.output is not None else None,
        goal=state.goal,
        lesson=lesson,
    )


def _validate_query(context_id: str, goal: str, limit: int) -> None:
    if not isinstance(context_id, str) or not context_id.strip():
        raise ValueError("context_id must be nonempty text")
    if not isinstance(goal, str):
        raise TypeError("goal must be text")
    if type(limit) is not int or not 0 <= limit <= MAX_RETRIEVAL_LIMIT:
        raise ValueError(f"limit must be an integer from 0 to {MAX_RETRIEVAL_LIMIT}")


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE))


def _score(query: str, goal: str, lesson: str) -> int:
    return len(_tokens(query) & _tokens(goal + " " + lesson))


class InMemoryExperienceStore:
    """First write wins for each task/context/call identity.

    Retrieval ranks all records in the requested context by unique word-token
    overlap, then by identifier. At most 100 records can be requested.
    """

    def __init__(self) -> None:
        self._records: dict[str, _Experience] = {}
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise ValueError("experience store is closed")

    def record(self, state: AgentState, result: ToolResult, lesson: str) -> str:
        self._ensure_open()
        experience = _make_experience(state, result, lesson)
        self._records.setdefault(experience.experience_id, experience)
        return experience.experience_id

    def retrieve(self, context_id: str, goal: str, limit: int = 5) -> list[dict]:
        self._ensure_open()
        _validate_query(context_id, goal, limit)
        if limit == 0:
            return []
        ranked = sorted(
            (record for record in self._records.values() if record.context_id == context_id),
            key=lambda record: (-_score(goal, record.goal, record.lesson), record.experience_id),
        )
        return [record.model_dump(mode="json") for record in ranked[:limit]]

    def close(self) -> None:
        self._closed = True
        self._records.clear()

    def __enter__(self) -> InMemoryExperienceStore:
        self._ensure_open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class SQLiteExperienceStore:
    """SQLite equivalent of InMemoryExperienceStore; not an authorization ledger.

    Use a database belonging to the application. No model credentials are
    required. The creating thread owns this connection and must close it.
    """

    def __init__(self, path: str | PathLike[str]) -> None:
        if not isinstance(path, (str, PathLike)) or not str(path).strip():
            raise ValueError("path must identify a SQLite database")
        self._connection: sqlite3.Connection | None = sqlite3.connect(path)
        try:
            self._connection.create_function("experience_overlap", 3, _score, deterministic=True)
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS agent_framework_experiences_v1 ("
                "experience_id TEXT PRIMARY KEY, context_id TEXT NOT NULL, "
                "goal TEXT NOT NULL, lesson TEXT NOT NULL, payload TEXT NOT NULL)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS agent_framework_experience_context_v1 "
                "ON agent_framework_experiences_v1(context_id)"
            )
            self._connection.commit()
        except Exception:
            self._connection.close()
            self._connection = None
            raise

    def _open_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise ValueError("experience store is closed")
        return self._connection

    def record(self, state: AgentState, result: ToolResult, lesson: str) -> str:
        connection = self._open_connection()
        experience = _make_experience(state, result, lesson)
        payload = json.dumps(experience.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, allow_nan=False)
        with connection:
            connection.execute(
                "INSERT OR IGNORE INTO agent_framework_experiences_v1 "
                "(experience_id, context_id, goal, lesson, payload) VALUES (?, ?, ?, ?, ?)",
                (experience.experience_id, experience.context_id, experience.goal, experience.lesson, payload),
            )
        return experience.experience_id

    def retrieve(self, context_id: str, goal: str, limit: int = 5) -> list[dict]:
        connection = self._open_connection()
        _validate_query(context_id, goal, limit)
        rows = connection.execute(
            "SELECT experience_id, payload FROM agent_framework_experiences_v1 "
            "WHERE context_id = ? "
            "ORDER BY experience_overlap(?, goal, lesson) DESC, experience_id ASC LIMIT ?",
            (context_id, goal, limit),
        ).fetchall()
        records = []
        for stored_id, payload in rows:
            record = _Experience.model_validate_json(payload)
            expected_id = fingerprint({
                "task_id": record.task_id,
                "context_id": record.context_id,
                "call_id": record.call_id,
            })
            if record.context_id != context_id or stored_id != record.experience_id or stored_id != expected_id:
                raise ValueError("experience record provenance is inconsistent")
            records.append(record.model_dump(mode="json"))
        return deepcopy(records)

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> SQLiteExperienceStore:
        self._open_connection()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
