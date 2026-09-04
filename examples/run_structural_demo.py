from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from domain_gated_mf.demo import AnalyticalDemoEvaluator, make_demo_records, make_demo_tasks, write_demo_summary  # noqa: E402
from domain_gated_mf.model import DomainGatedSurrogate  # noqa: E402
from domain_gated_mf.optimizer import DomainGatedOptimizer, SurrogateRuntime, UniformLocalCandidateGenerator  # noqa: E402
from domain_gated_mf.schema import AcquisitionConfig, GateThresholds, OptimizationConfig  # noqa: E402
from domain_gated_mf.training import TorchSurrogateTrainer, records_to_samples  # noqa: E402


def main() -> None:
    tasks = make_demo_tasks()
    evaluator = AnalyticalDemoEvaluator()
    torch.manual_seed(20260804)
    model = DomainGatedSurrogate(
        input_dim=10,
        latent_dim=8,
        response_dim=4,
        task_names=[task.name for task in tasks],
        hidden=(24,),
        adapter_hidden=(12,),
        field_coordinate_dim=1,
        field_dim=1,
    )
    trainer = TorchSurrogateTrainer(model, learning_rate=2.0e-3, lambda_phys=0.0, lambda_unc=0.2, lambda_aux=0.1)

    training_records = []
    calibration_records = []
    for index, task in enumerate(tasks):
        training_records.extend(make_demo_records(task, evaluator, 5, 100 + index, fidelity=task.fidelities[0].level))
        training_records.extend(make_demo_records(task, evaluator, 5, 200 + index, fidelity=task.fidelities[1].level))
        training_records.extend(make_demo_records(task, evaluator, 5, 300 + index, fidelity=task.fidelities[2].level))
        calibration_records.extend(make_demo_records(task, evaluator, 5, 400 + index, fidelity=task.fidelities[2].level))
    all_samples = []
    for task in tasks:
        all_samples.extend(records_to_samples(task, training_records))
    loss = trainer.fit(all_samples, epochs=2, seed=20260804)

    task = tasks[0]
    runtime = SurrogateRuntime(task, model, AcquisitionConfig(resamples=64, disagreement_weight=0.2))
    runtime.prepare(
        [record for record in training_records if record.task == task.name],
        [record for record in calibration_records if record.task == task.name],
        thresholds=GateThresholds(distance=0.0, width=0.0, convergence=1.0),
    )
    # Keep the archive empty so the strict-gate branch has a deterministic
    # positive conservative gain and demonstrates direct HF escalation.
    initial = ()
    result = DomainGatedOptimizer(
        task,
        runtime,
        evaluator,
        generator=UniformLocalCandidateGenerator(local_probability=0.5),
        config=OptimizationConfig(hf_budget=3, candidate_batch_size=3, max_batches=2, seed=77),
    ).run(initial_records=initial)
    summary = {
        "warning": "demonstration-only; not the published numerical results",
        "tasks": [item.name for item in tasks],
        "fidelities": [item.level.value for item in task.fidelities],
        "training_loss": loss.total,
        "events": len(result.events),
        "hf_calls": result.hf_calls,
        "archive_size": len(result.archive.records()),
    }
    write_demo_summary(ROOT / "demo_output" / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
