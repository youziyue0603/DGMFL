from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from .metrics import hypervolume_improvement
from .records import FidelityRepository
from .schema import AcquisitionConfig, Array, Fidelity, TaskSpec


@dataclass(frozen=True)
class CandidatePrediction:
    responses: Mapping[Fidelity, Array]
    hf_scale: Array
    convergence_probability: Mapping[Fidelity, float]
    feasibility_probability: Mapping[Fidelity, float]
    adapted_hf_prediction: Array
    shared_hf_prediction: Array
    adapted_representation: Array
    calibrated_hf_scale: Array | None = None

    def __post_init__(self) -> None:
        response_shapes = set()
        for level in Fidelity:
            value = np.asarray(self.responses[level], dtype=float)
            if value.ndim != 1 or not np.all(np.isfinite(value)):
                raise ValueError("Fidelity responses must be finite vectors.")
            response_shapes.add(value.shape)
            for name, mapping in (("convergence", self.convergence_probability), ("feasibility", self.feasibility_probability)):
                probability = float(mapping[level])
                if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
                    raise ValueError(f"{name} probability must lie in [0, 1].")
        if len(response_shapes) != 1:
            raise ValueError("All fidelity responses must have the same shape.")
        response_shape = next(iter(response_shapes))
        for name, value in (("HF scale", self.hf_scale), ("adapted HF prediction", self.adapted_hf_prediction), ("shared HF prediction", self.shared_hf_prediction)):
            array = np.asarray(value, dtype=float)
            if array.shape != response_shape or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must match the response-vector shape and be finite.")
        if self.calibrated_hf_scale is not None:
            calibrated = np.asarray(self.calibrated_hf_scale, dtype=float)
            if calibrated.shape != response_shape or not np.all(np.isfinite(calibrated)) or np.any(calibrated < 0.0):
                raise ValueError("Calibrated HF scale must match the response-vector shape and be nonnegative.")

    @property
    def acquisition_scale(self) -> Array:
        """Use the frozen calibrated scale when available for HV resampling."""
        return self.hf_scale if self.calibrated_hf_scale is None else self.calibrated_hf_scale


@dataclass(frozen=True)
class AllocationScore:
    fidelity: Fidelity
    conservative_gain: float
    convergence_probability: float
    feasibility_probability: float
    disagreement: float
    runtime: float
    score: float


class RuntimeEstimator:
    """Moving median with the task-interface runtime before enough samples exist."""

    def __init__(self, minimum_observations: int = 3) -> None:
        if minimum_observations < 1:
            raise ValueError("minimum_observations must be positive.")
        self.minimum_observations = minimum_observations
        self._priors: dict[tuple[str, Fidelity], float] = {}

    def set_calibration_priors(self, records: Iterable[object]) -> None:
        """Set calibration-mean runtimes used before enough online observations exist."""
        grouped: dict[tuple[str, Fidelity], list[float]] = {}
        for record in records:
            key = (record.task, record.fidelity)
            grouped.setdefault(key, []).append(float(record.runtime))
        self._priors = {key: float(np.mean(values)) for key, values in grouped.items() if values}

    def estimate(self, task: TaskSpec, fidelity: Fidelity, repository: FidelityRepository) -> float:
        observed = repository.runtimes(task.name, fidelity)
        nominal = task.fidelity_spec(fidelity).nominal_runtime
        if observed.size < self.minimum_observations:
            return float(self._priors.get((task.name, fidelity), nominal))
        return float(np.median(observed))


def residual_resampled_hv_gain(
    task: TaskSpec,
    existing_objectives: Array,
    response_mean: Array,
    response_scale: Array,
    rng: np.random.Generator,
    config: AcquisitionConfig,
) -> float:
    """Compute Q_0.25[Delta HV] using calibrated residual resampling."""
    mean = np.asarray(response_mean, dtype=float)
    scale = np.asarray(response_scale, dtype=float)
    existing = np.asarray(existing_objectives, dtype=float)
    if mean.ndim != 1 or scale.shape != mean.shape or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
        raise ValueError("Response mean and scale must be finite equal-shaped vectors.")
    if existing.ndim != 2 or existing.shape[1] != task.objective_dim:
        raise ValueError("Existing objective array has the wrong dimension.")
    if task.reference_point is None:
        raise ValueError("A task reference point is required for hypervolume gain.")
    samples = mean[None, :] + rng.normal(size=(config.resamples, mean.size)) * np.maximum(scale[None, :], 0.0)
    gains = np.asarray(
        [hypervolume_improvement(existing, task.objectives(response), task.reference_point) for response in samples],
        dtype=float,
    )
    return float(np.quantile(gains, 0.25, method="linear"))


class RiskCostFidelityAllocator:
    """Implements Eq. (acquisition) after the applicability gate has passed."""

    def __init__(self, config: AcquisitionConfig, runtime_estimator: RuntimeEstimator | None = None) -> None:
        self.config = config
        self.runtime_estimator = runtime_estimator or RuntimeEstimator(config.min_runtime_observations)

    def score(
        self,
        task: TaskSpec,
        prediction: CandidatePrediction,
        existing_objectives: Array,
        disagreement: float,
        repository: FidelityRepository,
        rng: np.random.Generator,
    ) -> tuple[AllocationScore, ...]:
        scores: list[AllocationScore] = []
        for fidelity in (Fidelity.L, Fidelity.M, Fidelity.H):
            gain = residual_resampled_hv_gain(
                task,
                existing_objectives,
                prediction.responses[fidelity],
                prediction.acquisition_scale,
                rng,
                self.config,
            )
            convergence = float(prediction.convergence_probability[fidelity])
            feasibility = float(prediction.feasibility_probability[fidelity])
            runtime = self.runtime_estimator.estimate(task, fidelity, repository)
            score = gain * convergence * feasibility * (1.0 + self.config.disagreement_weight * disagreement)
            score /= runtime + self.config.eps_time
            scores.append(AllocationScore(fidelity, gain, convergence, feasibility, disagreement, runtime, float(score)))
        return tuple(scores)

    @staticmethod
    def select(scores: tuple[AllocationScore, ...]) -> AllocationScore:
        if not scores:
            raise ValueError("No allocation scores supplied.")
        return max(scores, key=lambda item: (item.score, -list(Fidelity).index(item.fidelity)))

    def conservative_hf_gain(
        self,
        task: TaskSpec,
        prediction: CandidatePrediction,
        existing_objectives: Array,
        rng: np.random.Generator,
    ) -> float:
        return residual_resampled_hv_gain(
            task,
            existing_objectives,
            prediction.responses[Fidelity.H],
            prediction.acquisition_scale,
            rng,
            self.config,
        )


def disagreement_from_predictions(adapted_hf: Array, shared_hf: Array, epsilon: float = 1.0e-8) -> float:
    adapted = np.asarray(adapted_hf, dtype=float)
    shared = np.asarray(shared_hf, dtype=float)
    if adapted.shape != shared.shape or adapted.ndim != 1:
        raise ValueError("Adapted and shared predictions must be equal-shaped vectors.")
    return float(np.linalg.norm(adapted - shared) / (np.linalg.norm(adapted) + epsilon))
