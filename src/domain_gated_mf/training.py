from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .interfaces import PhysicsResidualProvider
from .model import DomainGatedSurrogate
from .records import EvaluationRecord
from .schema import Fidelity, TaskSpec


@dataclass(frozen=True)
class TrainingSample:
    task: TaskSpec
    fidelity: Fidelity
    descriptor: np.ndarray
    response: np.ndarray
    converged: bool
    feasible: bool
    field: np.ndarray | None = None
    field_coordinates: np.ndarray | None = None


@dataclass(frozen=True)
class LossBreakdown:
    total: float
    data: float
    energy: float
    equilibrium: float
    constraint: float
    uncertainty: float
    auxiliary: float


class ZeroPhysicsResiduals:
    """Default hook for tasks that expose scalar labels but no field residual."""

    def residual_loss(self, task, fidelity, descriptors, response_prediction, field_prediction):
        zero = response_prediction.sum() * 0.0
        return {"energy": zero, "equilibrium": zero, "constraint": zero}


def records_to_samples(task: TaskSpec, records: Iterable[EvaluationRecord]) -> tuple[TrainingSample, ...]:
    samples: list[TrainingSample] = []
    for record in records:
        if record.task != task.name:
            continue
        samples.append(
            TrainingSample(
                task=task,
                fidelity=record.fidelity,
                descriptor=task.descriptor(record.design),
                response=record.response,
                converged=record.converged,
                feasible=record.feasible,
                field=record.field,
                field_coordinates=record.field_coordinates,
            )
        )
    return tuple(samples)


class TorchSurrogateTrainer:
    """Optimization of Eq. (loss) with task residual hooks."""

    def __init__(
        self,
        model: DomainGatedSurrogate,
        learning_rate: float = 1.0e-3,
        lambda_phys: float = 1.0,
        lambda_unc: float = 1.0,
        lambda_aux: float = 0.1,
        device: str = "cpu",
    ) -> None:
        if min(learning_rate, lambda_phys, lambda_unc, lambda_aux) < 0.0:
            raise ValueError("Training weights must be nonnegative.")
        self.model = model.to(device)
        self.learning_rate = float(learning_rate)
        self.lambda_phys = float(lambda_phys)
        self.lambda_unc = float(lambda_unc)
        self.lambda_aux = float(lambda_aux)
        self.device = torch.device(device)

    def _constraint_loss(self, sample: TrainingSample, predicted_response: Tensor) -> Tensor:
        if sample.task.constraint_torch_fn is None:
            return predicted_response.sum() * 0.0
        predicted_constraints = sample.task.constraint_torch_fn(predicted_response)
        return F.relu(predicted_constraints).square().mean()

    def _sample_loss(self, sample: TrainingSample, physics: PhysicsResidualProvider) -> tuple[Tensor, LossBreakdown]:
        descriptor = torch.as_tensor(sample.descriptor, dtype=torch.float32, device=self.device).unsqueeze(0)
        target = torch.as_tensor(sample.response, dtype=torch.float32, device=self.device).unsqueeze(0)
        coordinates = None
        field_target = None
        if sample.field is not None and sample.field_coordinates is not None:
            coordinates = torch.as_tensor(sample.field_coordinates, dtype=torch.float32, device=self.device)
            field_target = torch.as_tensor(sample.field, dtype=torch.float32, device=self.device)
        output = self.model(descriptor, sample.task.name, field_coordinates=coordinates)
        predicted = output.responses[sample.fidelity]
        data = F.mse_loss(predicted, target)
        energy = predicted.sum() * 0.0
        equilibrium = predicted.sum() * 0.0
        constraint = self._constraint_loss(sample, predicted[0])
        field_prediction = None if output.fields is None else output.fields[sample.fidelity]
        residuals = physics.residual_loss(sample.task, sample.fidelity, descriptor, predicted, field_prediction)
        energy = residuals.get("energy", energy)
        equilibrium = residuals.get("equilibrium", equilibrium)
        constraint = constraint + residuals.get("constraint", constraint * 0.0)
        if field_prediction is not None and field_target is not None:
            equilibrium = equilibrium + F.mse_loss(field_prediction[0], field_target)
        uncertainty = predicted.sum() * 0.0
        if sample.fidelity is Fidelity.H:
            uncertainty = self.model.gaussian_nll(target, output.responses[Fidelity.H], output.hf_scale)
        conv_target = torch.as_tensor([float(sample.converged)], dtype=torch.float32, device=self.device)
        feas_target = torch.as_tensor([float(sample.feasible)], dtype=torch.float32, device=self.device)
        auxiliary = F.binary_cross_entropy_with_logits(
            output.convergence_logits[sample.fidelity], conv_target
        ) + F.binary_cross_entropy_with_logits(output.feasibility_logits[sample.fidelity], feas_target)
        total = data + self.lambda_phys * (energy + equilibrium + constraint) + self.lambda_unc * uncertainty + self.lambda_aux * auxiliary
        return total, LossBreakdown(
            float(total.detach().cpu()),
            float(data.detach().cpu()),
            float(energy.detach().cpu()),
            float(equilibrium.detach().cpu()),
            float(constraint.detach().cpu()),
            float(uncertainty.detach().cpu()),
            float(auxiliary.detach().cpu()),
        )

    def fit(
        self,
        samples: Iterable[TrainingSample],
        epochs: int = 50,
        seed: int = 20260804,
        physics_by_task: Mapping[str, PhysicsResidualProvider] | None = None,
    ) -> LossBreakdown:
        sample_list = tuple(samples)
        if not sample_list:
            raise ValueError("At least one training sample is required.")
        if epochs <= 0:
            raise ValueError("epochs must be positive.")
        torch.manual_seed(int(seed))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model.train()
        latest = None
        for _ in range(epochs):
            totals = []
            for sample in sample_list:
                optimizer.zero_grad(set_to_none=True)
                physics = (physics_by_task or {}).get(sample.task.name, ZeroPhysicsResiduals())
                total, breakdown = self._sample_loss(sample, physics)
                total.backward()
                optimizer.step()
                totals.append(breakdown)
            latest = LossBreakdown(
                *(float(np.mean([getattr(item, field) for item in totals])) for field in LossBreakdown.__dataclass_fields__)
            )
        return latest

