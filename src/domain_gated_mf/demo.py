from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .interfaces import EvaluationResult
from .records import EvaluationRecord
from .schema import Fidelity, FidelitySpec, TaskSpec


def make_demo_tasks() -> tuple[TaskSpec, ...]:
    """Six deterministic analytical interfaces for software testing only."""
    tasks: list[TaskSpec] = []
    for index, name in enumerate(("LCE", "SMP", "HG", "DEA", "MRC", "PRF")):
        def descriptor(design, task_index=index):
            value = np.asarray(design, dtype=float)
            normalized = (value - 0.0) / 1.0
            one_hot = np.zeros(6, dtype=float)
            one_hot[task_index] = 1.0
            return np.concatenate((normalized, one_hot))

        def response_to_objectives(response):
            return np.asarray(response, dtype=float)[:3]

        def response_to_constraints(response):
            return np.asarray(response, dtype=float)[3:4]

        def response_to_constraints_torch(response):
            return response[..., 3:4]

        tasks.append(
            TaskSpec(
                name=name,
                design_lower=np.zeros(4),
                design_upper=np.ones(4),
                descriptor_fn=descriptor,
                objective_fn=response_to_objectives,
                constraint_fn=response_to_constraints,
                response_dim=4,
                objective_dim=3,
                constraint_dim=1,
                constraint_torch_fn=response_to_constraints_torch,
                fidelities=(
                    FidelitySpec(Fidelity.L, "LF", "analytical screening approximation", 1.0),
                    FidelitySpec(Fidelity.M, "MF", "reduced coupled response approximation", 2.0),
                    FidelitySpec(Fidelity.H, "HF", "reference analytical interface", 4.0),
                ),
                reference_point=np.asarray([1.8, 1.8, 1.8]),
            )
        )
    return tuple(tasks)


class AnalyticalDemoEvaluator:
    """Toy evaluator with the same result contract as a real task solver."""

    def evaluate(self, task: TaskSpec, design: np.ndarray, fidelity: Fidelity) -> EvaluationResult:
        x = task.validate_design(design)
        index = ("LCE", "SMP", "HG", "DEA", "MRC", "PRF").index(task.name)
        center = np.asarray(
            [0.20 + 0.10 * ((index + 1) % 3), 0.25 + 0.07 * (index % 3), 0.35 + 0.06 * (index % 2), 0.65 - 0.05 * (index % 3)],
            dtype=float,
        )
        shift = {Fidelity.L: 0.08, Fidelity.M: 0.03, Fidelity.H: 0.0}[fidelity]
        objective = np.asarray(
            [
                float(np.mean((x - center) ** 2) + 0.08 + shift),
                float(np.mean((x - np.roll(center, 1)) ** 2) + 0.08 + 0.5 * shift),
                float(np.mean((x - np.roll(center, 2)) ** 2) + 0.08 + 0.25 * shift),
            ]
        )
        constraint = float(np.sum(x) - 3.15)
        response = np.concatenate((objective, np.asarray([constraint])))
        coordinates = np.linspace(0.0, 1.0, 12)[:, None]
        field = (np.sin(2.0 * np.pi * coordinates[:, 0]) * (1.0 - objective[0]))[:, None]
        return EvaluationResult(
            response=response,
            converged=bool(fidelity is not Fidelity.H or np.sum(x) < 3.9),
            feasible=bool(constraint <= 0.0),
            runtime=float(task.fidelity_spec(fidelity).nominal_runtime * (1.0 + 0.1 * np.mean(x))),
            field=field,
            field_coordinates=coordinates,
            metadata={"demo_only": True, "task_index": index},
        )


def make_demo_records(task: TaskSpec, evaluator: AnalyticalDemoEvaluator, count: int, seed: int, fidelity: Fidelity) -> tuple[EvaluationRecord, ...]:
    rng = np.random.default_rng(seed)
    records = []
    for design in rng.random((count, task.design_dim)):
        result = evaluator.evaluate(task, design, fidelity)
        records.append(EvaluationRecord.from_result(task, design, fidelity, result))
    return tuple(records)


def write_demo_summary(path: str | Path, summary: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
