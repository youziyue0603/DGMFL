from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .schema import FIDELITIES, Fidelity


def _mlp(input_dim: int, output_dim: int, hidden: tuple[int, ...], activation: type[nn.Module] = nn.Tanh) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for width in hidden:
        layers.extend((nn.Linear(current, width), activation()))
        current = width
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class TaskAdapter(nn.Module):
    def __init__(self, latent_dim: int, hidden: tuple[int, ...]) -> None:
        super().__init__()
        self.delta = _mlp(latent_dim, latent_dim, hidden)

    def forward(self, shared_latent: Tensor) -> Tensor:
        return shared_latent + self.delta(shared_latent)


class FieldOperator(nn.Module):
    def __init__(self, latent_dim: int, coordinate_dim: int, field_dim: int, hidden: tuple[int, ...]) -> None:
        super().__init__()
        self.coordinate_dim = coordinate_dim
        self.field_dim = field_dim
        self.network = _mlp(latent_dim + coordinate_dim, field_dim, hidden)

    def forward(self, latent: Tensor, coordinates: Tensor) -> Tensor:
        if coordinates.ndim == 2:
            coordinates = coordinates.unsqueeze(0).expand(latent.shape[0], -1, -1)
        if coordinates.ndim != 3 or coordinates.shape[0] != latent.shape[0] or coordinates.shape[-1] != self.coordinate_dim:
            raise ValueError("Coordinates must have shape (batch, points, coordinate_dim).")
        expanded = latent.unsqueeze(1).expand(-1, coordinates.shape[1], -1)
        return self.network(torch.cat((expanded, coordinates), dim=-1))


@dataclass(frozen=True)
class ModelOutput:
    shared_latent: Tensor
    adapted_latent: Tensor
    responses: Mapping[Fidelity, Tensor]
    hf_scale: Tensor
    convergence_logits: Mapping[Fidelity, Tensor]
    feasibility_logits: Mapping[Fidelity, Tensor]
    fields: Mapping[Fidelity, Tensor] | None = None


class DomainGatedSurrogate(nn.Module):
    """Shared/task-specific residual surrogate in the final 2.0 method."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        response_dim: int,
        task_names: Iterable[str],
        hidden: tuple[int, ...] = (64, 64),
        adapter_hidden: tuple[int, ...] = (32,),
        field_coordinate_dim: int | None = None,
        field_dim: int = 0,
        sigma_min: float = 1.0e-6,
    ) -> None:
        super().__init__()
        task_names_value = tuple(task_names)
        if not task_names_value:
            raise ValueError("At least one task name is required.")
        if min(input_dim, latent_dim, response_dim) <= 0:
            raise ValueError("Model dimensions must be positive.")
        if sigma_min <= 0.0:
            raise ValueError("sigma_min must be positive.")
        if (field_coordinate_dim is None) != (field_dim == 0):
            raise ValueError("Set both field_coordinate_dim and field_dim for field supervision, or neither.")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.response_dim = response_dim
        self.task_names = task_names_value
        self.field_coordinate_dim = field_coordinate_dim
        self.field_dim = field_dim
        self.sigma_min = float(sigma_min)

        self.shared_encoder = _mlp(input_dim, latent_dim, hidden)
        self.adapters = nn.ModuleDict({name: TaskAdapter(latent_dim, adapter_hidden) for name in task_names_value})
        self.lf_head = _mlp(latent_dim, response_dim, hidden)
        self.mf_residual_head = _mlp(latent_dim, response_dim, hidden)
        self.hf_residual_head = _mlp(latent_dim, response_dim, hidden)
        self.scale_head = _mlp(latent_dim, response_dim, hidden)
        self.convergence_heads = nn.ModuleDict({level.value: _mlp(latent_dim, 1, hidden) for level in FIDELITIES})
        self.feasibility_heads = nn.ModuleDict({level.value: _mlp(latent_dim, 1, hidden) for level in FIDELITIES})
        self.field_operators = (
            nn.ModuleDict(
                {
                    level.value: FieldOperator(latent_dim, field_coordinate_dim, field_dim, hidden)
                    for level in FIDELITIES
                }
            )
            if field_coordinate_dim is not None
            else None
        )

    def _check_task(self, task_name: str) -> None:
        if task_name not in self.adapters:
            raise KeyError(f"Unknown task {task_name!r}; registered tasks are {self.task_names}.")

    def encode(self, descriptors: Tensor, task_name: str, adapted: bool = True) -> tuple[Tensor, Tensor]:
        self._check_task(task_name)
        if descriptors.ndim != 2 or descriptors.shape[-1] != self.input_dim:
            raise ValueError(f"Descriptors must have shape (batch, {self.input_dim}).")
        shared = self.shared_encoder(descriptors)
        adapted_latent = self.adapters[task_name](shared) if adapted else shared
        return shared, adapted_latent

    def _responses(self, latent: Tensor) -> dict[Fidelity, Tensor]:
        lf = self.lf_head(latent)
        mf = lf + self.mf_residual_head(latent)
        hf = mf + self.hf_residual_head(latent)
        return {Fidelity.L: lf, Fidelity.M: mf, Fidelity.H: hf}

    def forward(
        self,
        descriptors: Tensor,
        task_name: str,
        field_coordinates: Tensor | None = None,
        adapted: bool = True,
    ) -> ModelOutput:
        shared, adapted_latent = self.encode(descriptors, task_name, adapted=adapted)
        responses = self._responses(adapted_latent)
        scale = F.softplus(self.scale_head(adapted_latent)) + self.sigma_min
        convergence_logits = {level: self.convergence_heads[level.value](adapted_latent).squeeze(-1) for level in FIDELITIES}
        feasibility_logits = {level: self.feasibility_heads[level.value](adapted_latent).squeeze(-1) for level in FIDELITIES}
        fields = None
        if field_coordinates is not None:
            if self.field_operators is None:
                raise ValueError("This model was constructed without field operators.")
            fields = {level: self.field_operators[level.value](adapted_latent, field_coordinates) for level in FIDELITIES}
        return ModelOutput(shared, adapted_latent, responses, scale, convergence_logits, feasibility_logits, fields)

    @torch.no_grad()
    def predict_numpy(self, descriptors: np.ndarray, task_name: str) -> ModelOutput:
        value = np.asarray(descriptors, dtype=np.float32)
        was_single = value.ndim == 1
        if was_single:
            value = value[None, :]
        output = self(torch.as_tensor(value), task_name)
        if not was_single:
            return output
        return ModelOutput(
            output.shared_latent[0],
            output.adapted_latent[0],
            {level: tensor[0] for level, tensor in output.responses.items()},
            output.hf_scale[0],
            {level: tensor[0] for level, tensor in output.convergence_logits.items()},
            {level: tensor[0] for level, tensor in output.feasibility_logits.items()},
            output.fields,
        )

    @torch.no_grad()
    def shared_and_adapted_hf_numpy(self, descriptors: np.ndarray, task_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        value = np.asarray(descriptors, dtype=np.float32)
        if value.ndim == 1:
            value = value[None, :]
        adapted = self(torch.as_tensor(value), task_name, adapted=True)
        shared = self(torch.as_tensor(value), task_name, adapted=False)
        return (
            adapted.responses[Fidelity.H].detach().cpu().numpy(),
            shared.responses[Fidelity.H].detach().cpu().numpy(),
            adapted.adapted_latent.detach().cpu().numpy(),
        )

    def gaussian_nll(self, target: Tensor, mean: Tensor, scale: Tensor) -> Tensor:
        variance = scale.square()
        return 0.5 * torch.mean(((target - mean).square() / variance) + 2.0 * torch.log(scale))

