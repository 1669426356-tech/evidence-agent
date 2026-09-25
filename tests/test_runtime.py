from __future__ import annotations

import json

import pytest

from agent_framework.runtime import load_state, save_state
from agent_framework.state import AgentState


def test_state_snapshot_roundtrip_preserves_identity_and_evidence(tmp_path):
    state = AgentState(context_id="example", goal="Inspect", metadata={"attempt": 1})
    path = tmp_path / "snapshots" / "state.json"
    save_state(state, path)
    restored = load_state(path)
    assert restored == state
    assert not list(path.parent.glob("*.tmp"))


@pytest.mark.parametrize("mutate", [
    lambda payload: payload.update(schema_version=999),
    lambda payload: payload.update(schema_version=True),
    lambda payload: payload.update(unexpected="extra field"),
    lambda payload: payload["state"].update(approved=True),
])
def test_snapshot_rejects_unknown_version_and_extra_fields(tmp_path, mutate):
    state = AgentState(context_id="example", goal="Inspect")
    path = tmp_path / "state.json"
    save_state(state, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_state(path)
