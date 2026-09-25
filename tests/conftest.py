from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ValueInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int = 1


class EmptyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

