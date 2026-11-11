from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DomainError(Exception):
    """Stable code only; private input must not be placed in a public error."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Case(StrEnum):
    PHYSICS = "physics"
    HOME = "private_home"


class Record(Contract):
    id: str
    case: Case
    group: str
    time_index: int
    context: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str | None = None
    rubric: str | None = None
    figure_descriptions: list[str] = Field(default_factory=list)
    points: float = Field(default=1.0, gt=0)
    mean_score_rate: float | None = Field(default=None, ge=0, le=1)
    choice: int | None = Field(default=None, ge=0, le=2)
    question_type: Literal["choice", "written", "scene"]

    @model_validator(mode="after")
    def target_matches_task(self):
        if self.case == Case.PHYSICS and (self.mean_score_rate is None or self.choice is not None):
            raise ValueError("physics_mean_target_required")
        if self.case == Case.HOME and (self.choice is None or self.mean_score_rate is not None):
            raise ValueError("home_choice_target_required")
        return self

    def prompt(self) -> str:
        import json

        return json.dumps(
            {
                "task": self.case.value,
                "context": self.context,
                "question": self.question,
                "answer": self.answer,
                "rubric": self.rubric,
                "figures": self.figure_descriptions,
            },
            ensure_ascii=False,
            sort_keys=True,
        )


class DatasetRequest(Contract):
    dataset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,79}$")


class SplitRequest(DatasetRequest):
    rule_id: Literal["whole-group-time-v1"] = "whole-group-time-v1"


class RunRequest(DatasetRequest):
    split_id: str = Field(pattern=r"^[a-f0-9]{24}$")
    recipe_id: Literal["cpu-head-v1", "cpu-lora-v1"]
    seed: int = Field(default=7, ge=0, le=2147483647)
    max_steps: int = Field(default=8, ge=1, le=100)
    max_seconds: int = Field(default=120, ge=1, le=600)


class ComparisonRequest(Contract):
    run_ids: list[str] = Field(min_length=2, max_length=8)


class ShadowApproval(Contract):
    note: str = Field(min_length=1, max_length=512)


class HomeIntent(StrEnum):
    READING = "reading"
    ENTRY = "open_entry"
    AMBIGUOUS = "ambiguous"


class HomeRequest(Contract):
    intent: HomeIntent
    space: Literal["study", "guest_room"] = "study"
    model_online: bool = True
    human_confirmed: bool = False


class Metric(Contract):
    name: str
    value: float


class GroupDiagnostic(Contract):
    group: str
    count: int
    metrics: list[Metric]


class Diagnostics(Contract):
    run_id: str
    status: str
    case: Case
    evidence_level: Literal["synthetic_cpu_smoke"]
    dataset_version: str
    split_id: str
    recipe_id: str
    code_version: str
    model_id: str
    objective: str
    validation_count: int
    independent_validation_groups: int
    steps: int
    train_curve: list[float]
    validation: list[Metric]
    groups: list[GroupDiagnostic]
    suppressed_groups: int
    resource: dict[str, int | float]
    error_code: str | None = None
    claims: list[str]
