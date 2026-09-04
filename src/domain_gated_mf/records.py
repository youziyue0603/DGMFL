from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Iterable

import numpy as np

from .interfaces import EvaluationResult
from .schema import Array, Fidelity, TaskSpec


def design_key(design: Array, decimals: int = 12) -> tuple[float, ...]:
    value = np.asarray(design, dtype=float)
    return tuple(np.round(value, decimals=decimals).tolist())


@dataclass(frozen=True)
class EvaluationRecord:
    task: str
    design: Array
    fidelity: Fidelity
    response: Array
    converged: bool
    feasible: bool
    runtime: float
    field: Array | None = None
    field_coordinates: Array | None = None
    objectives: Array | None = None
    constraints: Array | None = None
    metadata: dict[str, str | float | int | bool] = dataclass_field(default_factory=dict)

    @classmethod
    def from_result(
        cls,
        task: TaskSpec,
        design: Array,
        fidelity: Fidelity,
        result: EvaluationResult,
    ) -> "EvaluationRecord":
        response = np.asarray(result.response, dtype=float)
        if response.shape != (task.response_dim,):
            raise ValueError(f"Expected response shape {(task.response_dim,)}, got {response.shape}.")
        objectives = task.objectives(response)
        constraints = task.constraints(response)
        return cls(
            task=task.name,
            design=task.validate_design(design).copy(),
            fidelity=fidelity,
            response=response.copy(),
            converged=bool(result.converged),
            feasible=bool(result.feasible and np.all(constraints <= 0.0)),
            runtime=float(result.runtime),
            field=None if result.field is None else np.asarray(result.field, dtype=float).copy(),
            field_coordinates=None
            if result.field_coordinates is None
            else np.asarray(result.field_coordinates, dtype=float).copy(),
            objectives=objectives.copy(),
            constraints=constraints.copy(),
            metadata=dict(result.metadata),
        )


class FidelityRepository:
    """All observed data used by the surrogate and reliability calculations."""

    def __init__(self) -> None:
        self._records: list[EvaluationRecord] = []
        self._latest: dict[tuple[str, tuple[float, ...], Fidelity], EvaluationRecord] = {}

    def add(self, record: EvaluationRecord) -> None:
        key = (record.task, design_key(record.design), record.fidelity)
        self._records.append(record)
        self._latest[key] = record

    def extend(self, records: Iterable[EvaluationRecord]) -> None:
        for record in records:
            self.add(record)

    def all(self) -> tuple[EvaluationRecord, ...]:
        return tuple(self._records)

    def latest(self) -> tuple[EvaluationRecord, ...]:
        return tuple(self._latest.values())

    def for_task(self, task: str) -> tuple[EvaluationRecord, ...]:
        return tuple(record for record in self._records if record.task == task)

    def at_fidelity(self, task: str, fidelity: Fidelity) -> tuple[EvaluationRecord, ...]:
        return tuple(record for record in self._records if record.task == task and record.fidelity == fidelity)

    def get(self, task: str, design: Array, fidelity: Fidelity) -> EvaluationRecord | None:
        return self._latest.get((task, design_key(design), fidelity))

    def runtimes(self, task: str, fidelity: Fidelity) -> Array:
        return np.asarray([r.runtime for r in self.at_fidelity(task, fidelity)], dtype=float)


class WorkingCandidatePool:
    """Search state that may contain LF/MF evidence, unlike the report archive."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, tuple[float, ...]], EvaluationRecord] = {}

    def add(self, record: EvaluationRecord) -> None:
        self._records[(record.task, design_key(record.design))] = record

    def extend(self, records: Iterable[EvaluationRecord]) -> None:
        for record in records:
            self.add(record)

    def designs(self, task: str) -> Array:
        values = [record.design for (name, _), record in self._records.items() if name == task]
        if not values:
            return np.empty((0, 0), dtype=float)
        return np.vstack(values)

    def records(self, task: str | None = None) -> tuple[EvaluationRecord, ...]:
        if task is None:
            return tuple(self._records.values())
        return tuple(record for record in self._records.values() if record.task == task)


def _is_nondominated(points: Array) -> Array:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2:
        raise ValueError("Objective points must be a two-dimensional array.")
    keep = np.ones(values.shape[0], dtype=bool)
    for i, point in enumerate(values):
        if not keep[i]:
            continue
        dominates = np.all(values <= point, axis=1) & np.any(values < point, axis=1)
        if np.any(dominates):
            keep[i] = False
            continue
        equal = np.all(np.isclose(values, point, rtol=0.0, atol=1.0e-12), axis=1)
        first_equal = np.flatnonzero(equal)[0]
        if first_equal != i:
            keep[i] = False
    return keep


class ReportedParetoArchive:
    """HF-only archive admission rule from the manuscript."""

    def __init__(self, task: TaskSpec) -> None:
        self.task = task
        self._records: list[EvaluationRecord] = []

    def try_add(self, record: EvaluationRecord) -> bool:
        if record.task != self.task.name or record.fidelity is not Fidelity.H:
            return False
        if not record.converged or not record.feasible or record.objectives is None:
            return False
        candidate = list(self._records) + [record]
        objectives = np.vstack([item.objectives for item in candidate])
        keep = _is_nondominated(objectives)
        self._records = [item for item, selected in zip(candidate, keep) if selected]
        return any(item is record for item in self._records)

    def records(self) -> tuple[EvaluationRecord, ...]:
        return tuple(self._records)

    def objectives(self) -> Array:
        if not self._records:
            return np.empty((0, self.task.objective_dim), dtype=float)
        return np.vstack([record.objectives for record in self._records])
