from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Protocol

import numpy as np

from .schema import Array, Fidelity, TaskSpec


@dataclass(frozen=True)
class EvaluationResult:
    response: Array
    converged: bool
    feasible: bool
    runtime: float
    field: Array | None = None
    field_coordinates: Array | None = None
    metadata: dict[str, str | float | int | bool] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        response = np.asarray(self.response, dtype=float)
        if response.ndim != 1 or not np.all(np.isfinite(response)):
            raise ValueError("Evaluation response must be finite and one-dimensional.")
        if not np.isfinite(self.runtime) or self.runtime <= 0.0:
            raise ValueError("Evaluation runtime must be finite and positive.")
        if self.field is not None and not np.all(np.isfinite(np.asarray(self.field, dtype=float))):
            raise ValueError("Field values must be finite when supplied.")
        if self.field_coordinates is not None and not np.all(np.isfinite(np.asarray(self.field_coordinates, dtype=float))):
            raise ValueError("Field coordinates must be finite when supplied.")


class FidelityEvaluator(Protocol):
    def evaluate(self, task: TaskSpec, design: Array, fidelity: Fidelity) -> EvaluationResult:
        """Evaluate one candidate through the fixed task-specific interface."""


class PhysicsResidualProvider(Protocol):
    def residual_loss(
        self,
        task: TaskSpec,
        fidelity: Fidelity,
        descriptors: "object",
        response_prediction: "object",
        field_prediction: "object | None",
    ) -> dict[str, "object"]:
        """Return optional energy, equilibrium and constraint residual losses."""


class CandidateGenerator(Protocol):
    def generate(
        self,
        task: TaskSpec,
        working_designs: Array,
        count: int,
        rng: np.random.Generator,
    ) -> Array:
        """Generate candidates from the current search state."""


class SurrogateTrainer(Protocol):
    def update(self, task: TaskSpec, records: object) -> None:
        """Update the surrogate after new fidelity-specific observations."""
