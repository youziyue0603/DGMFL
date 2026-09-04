from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping

import numpy as np


class Fidelity(str, Enum):
    L = "L"
    M = "M"
    H = "H"


FIDELITIES: tuple[Fidelity, ...] = (Fidelity.L, Fidelity.M, Fidelity.H)
TASK_FAMILIES: tuple[str, ...] = ("LCE", "SMP", "HG", "DEA", "MRC", "PRF")


Array = np.ndarray
DescriptorFn = Callable[[Array], Array]
ObjectiveFn = Callable[[Array], Array]
ConstraintFn = Callable[[Array], Array]


@dataclass(frozen=True)
class FidelitySpec:
    level: Fidelity
    label: str
    description: str
    nominal_runtime: float


@dataclass(frozen=True)
class TaskSpec:
    """Numerical interface for one material family.

    The manuscript deliberately leaves the physical solver behind this
    interface. All functions operate on one-dimensional arrays and must be
    deterministic for a fixed input and fidelity. Objective projections must
    use the minimization direction assumed by the manuscript HV definition.
    """

    name: str
    design_lower: Array
    design_upper: Array
    descriptor_fn: DescriptorFn
    objective_fn: ObjectiveFn
    constraint_fn: ConstraintFn
    response_dim: int
    objective_dim: int
    constraint_dim: int
    constraint_torch_fn: Callable[[object], object] | None = None
    fidelities: tuple[FidelitySpec, ...] = field(default_factory=tuple)
    reference_point: Array | None = None

    def __post_init__(self) -> None:
        lower = np.asarray(self.design_lower, dtype=float)
        upper = np.asarray(self.design_upper, dtype=float)
        if lower.ndim != 1 or upper.shape != lower.shape:
            raise ValueError("Design bounds must be one-dimensional arrays of equal shape.")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or np.any(upper <= lower):
            raise ValueError("Design bounds must be finite and strictly increasing.")
        if not self.fidelities:
            object.__setattr__(self, "fidelities", default_fidelities())
        levels = tuple(spec.level for spec in self.fidelities)
        if levels != FIDELITIES:
            raise ValueError(f"Fidelity order must be {FIDELITIES}, got {levels}.")
        if self.reference_point is not None:
            ref = np.asarray(self.reference_point, dtype=float)
            if ref.shape != (self.objective_dim,):
                raise ValueError("Reference point dimension does not match objective_dim.")

    @property
    def design_dim(self) -> int:
        return int(np.asarray(self.design_lower).size)

    def validate_design(self, design: Array) -> Array:
        value = np.asarray(design, dtype=float)
        if value.shape != (self.design_dim,):
            raise ValueError(f"Expected design shape {(self.design_dim,)}, got {value.shape}.")
        if not np.all(np.isfinite(value)):
            raise ValueError("Design must contain finite values.")
        return value

    def descriptor(self, design: Array) -> Array:
        value = np.asarray(self.descriptor_fn(self.validate_design(design)), dtype=float)
        if value.ndim != 1 or not np.all(np.isfinite(value)):
            raise ValueError("Descriptor function must return a finite one-dimensional array.")
        return value

    def objectives(self, response: Array) -> Array:
        value = np.asarray(self.objective_fn(np.asarray(response, dtype=float)), dtype=float)
        if value.shape != (self.objective_dim,) or not np.all(np.isfinite(value)):
            raise ValueError("Objective function returned an invalid vector.")
        return value

    def constraints(self, response: Array) -> Array:
        value = np.asarray(self.constraint_fn(np.asarray(response, dtype=float)), dtype=float)
        if value.shape != (self.constraint_dim,) or not np.all(np.isfinite(value)):
            raise ValueError("Constraint function returned an invalid vector.")
        return value

    def fidelity_spec(self, fidelity: Fidelity) -> FidelitySpec:
        for spec in self.fidelities:
            if spec.level == fidelity:
                return spec
        raise KeyError(fidelity)


@dataclass(frozen=True)
class GateThresholds:
    distance: float
    width: float
    convergence: float

    def __post_init__(self) -> None:
        values = np.asarray([self.distance, self.width, self.convergence], dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("Gate thresholds must be finite and nonnegative.")
        if self.convergence > 1.0:
            raise ValueError("Convergence threshold cannot exceed one.")


@dataclass(frozen=True)
class AcquisitionConfig:
    alpha: float = 0.10
    resamples: int = 64
    disagreement_weight: float = 0.20
    eps_time: float = 1.0e-8
    eps_uncertainty: float = 1.0e-8
    eps_width: float = 1.0e-8
    min_runtime_observations: int = 3

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0 or self.resamples < 1:
            raise ValueError("Invalid uncertainty configuration.")
        if self.disagreement_weight < 0.0:
            raise ValueError("Disagreement weight must be nonnegative.")


@dataclass(frozen=True)
class OptimizationConfig:
    hf_budget: int = 120
    candidate_batch_size: int = 16
    max_batches: int = 1000
    initial_pool_size: int = 32
    seed: int = 20260804

    def __post_init__(self) -> None:
        if self.hf_budget <= 0 or self.candidate_batch_size <= 0 or self.max_batches <= 0:
            raise ValueError("Optimization counts must be positive.")


def default_fidelities() -> tuple[FidelitySpec, ...]:
    return (
        FidelitySpec(Fidelity.L, "LF", "cheapest task-specific screening interface", 1.0),
        FidelitySpec(Fidelity.M, "MF", "intermediate task-specific mechanics interface", 4.0),
        FidelitySpec(Fidelity.H, "HF", "reference mechanics interface for verification", 16.0),
    )


def normalise_design(design: Array, task: TaskSpec) -> Array:
    value = task.validate_design(design)
    lower = np.asarray(task.design_lower, dtype=float)
    upper = np.asarray(task.design_upper, dtype=float)
    return (value - lower) / (upper - lower)


def ensure_common_descriptor(tasks: Mapping[str, TaskSpec]) -> int:
    dimensions = {task.descriptor(np.asarray(task.design_lower, dtype=float)).size for task in tasks.values()}
    if len(dimensions) != 1:
        raise ValueError(f"All tasks must expose one common descriptor dimension, got {dimensions}.")
    return dimensions.pop()
