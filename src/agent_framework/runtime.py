"""Inspectable state snapshots. Loading a snapshot never resumes execution."""

from __future__ import annotations

import json
from pathlib import Path
import os
import tempfile

from .state import AgentState


def save_state(state: AgentState, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    validated = AgentState.model_validate(state.model_dump(mode="json"))
    payload = {"schema_version": 1, "state": validated.model_dump(mode="json")}
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                         prefix=f".{target.name}.", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary and Path(temporary).exists():
            Path(temporary).unlink()
    return target


def load_state(path: str | Path) -> AgentState:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(payload, dict) or set(payload) != {"schema_version", "state"}
            or type(payload["schema_version"]) is not int or payload["schema_version"] != 1):
        raise ValueError("Unsupported state snapshot schema")
    return AgentState.model_validate(payload["state"])
