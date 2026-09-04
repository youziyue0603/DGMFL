from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import FrozenMarginalCalibration
from .schema import Array, GateThresholds


@dataclass(frozen=True)
class ReferenceLatentCloud:
    """Training latent cloud and regularized covariance in Eq. (distance)."""

    representations: Array
    mean: Array
    covariance: Array
    inverse_covariance: Array
    regularization: float

    @classmethod
    def fit(cls, representations: Array, regularization: float = 1.0e-6) -> "ReferenceLatentCloud":
        values = np.asarray(representations, dtype=float)
        if values.ndim != 2 or values.shape[0] < 2 or not np.all(np.isfinite(values)):
            raise ValueError("Reference representations must be a finite N x D array with N >= 2.")
        if regularization <= 0.0:
            raise ValueError("Covariance regularization must be positive.")
        mean = np.mean(values, axis=0)
        centered = values - mean
        covariance = np.cov(centered, rowvar=False, ddof=1)
        covariance = np.atleast_2d(covariance) + regularization * np.eye(values.shape[1])
        inverse = np.linalg.pinv(covariance)
        return cls(values.copy(), mean, covariance, inverse, float(regularization))

    def distance(self, representation: Array) -> float | Array:
        value = np.asarray(representation, dtype=float)
        if value.shape[-1] != self.mean.size:
            raise ValueError("Representation dimension does not match reference cloud.")
        delta = value[None, :, :] - self.representations[:, None, :] if value.ndim == 2 else value - self.representations
        if value.ndim == 1:
            distances = np.einsum("nd,de,ne->n", delta, self.inverse_covariance, delta)
            return float(np.sqrt(max(np.min(distances), 0.0)))
        distances = np.einsum("nbd,de,nbe->nb", delta, self.inverse_covariance, delta)
        return np.sqrt(np.maximum(np.min(distances, axis=0), 0.0))


@dataclass(frozen=True)
class GateEvidence:
    distance: float
    normalized_width: float
    convergence_probability: float


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    evidence: GateEvidence
    reasons: tuple[str, ...]


def thresholds_from_calibration(
    distances: Array,
    normalized_widths: Array,
    convergence_probabilities: Array,
    distance_quantile: float = 0.95,
    width_quantile: float = 0.90,
    convergence_quantile: float = 0.05,
    converged_labels: Array | None = None,
) -> GateThresholds:
    distances_value = np.asarray(distances, dtype=float).reshape(-1)
    widths_value = np.asarray(normalized_widths, dtype=float).reshape(-1)
    convergence_value = np.asarray(convergence_probabilities, dtype=float).reshape(-1)
    if any(array.size == 0 for array in (distances_value, widths_value, convergence_value)):
        raise ValueError("Calibration evidence cannot be empty.")
    if not all(np.all(np.isfinite(array)) for array in (distances_value, widths_value, convergence_value)):
        raise ValueError("Calibration evidence must be finite.")
    if converged_labels is None:
        converged = convergence_value[convergence_value > 0.5]
    else:
        labels = np.asarray(converged_labels, dtype=bool).reshape(-1)
        if labels.shape != convergence_value.shape:
            raise ValueError("Convergence labels must match calibration evidence shape.")
        converged = convergence_value[labels]
    source = converged if converged.size else convergence_value
    return GateThresholds(
        float(np.quantile(distances_value, distance_quantile, method="higher")),
        float(np.quantile(widths_value, width_quantile, method="higher")),
        float(np.quantile(source, convergence_quantile, method="higher")),
    )


class ApplicabilityDomainGate:
    """Gate LF/MF search evidence; it is not an HF feasibility certificate."""

    def __init__(self, thresholds: GateThresholds) -> None:
        self.thresholds = thresholds

    def decide(self, evidence: GateEvidence) -> GateDecision:
        values = np.asarray(
            [evidence.distance, evidence.normalized_width, evidence.convergence_probability],
            dtype=float,
        )
        reasons: list[str] = []
        if not np.all(np.isfinite(values)):
            return GateDecision(False, evidence, ("nonfinite evidence",))
        if evidence.distance > self.thresholds.distance:
            reasons.append("latent distance exceeds threshold")
        if evidence.normalized_width > self.thresholds.width:
            reasons.append("calibrated interval width exceeds threshold")
        if evidence.convergence_probability < self.thresholds.convergence:
            reasons.append("HF convergence probability is below threshold")
        return GateDecision(not reasons, evidence, tuple(reasons))


def make_gate_evidence(
    cloud: ReferenceLatentCloud,
    adapted_representation: Array,
    hf_scale: Array,
    calibration: FrozenMarginalCalibration,
    hf_convergence_probability: float,
) -> GateEvidence:
    width_vector = np.asarray(calibration.normalized_half_width(np.asarray(hf_scale, dtype=float)))
    width = float(np.max(width_vector))
    return GateEvidence(
        distance=float(cloud.distance(np.asarray(adapted_representation, dtype=float))),
        normalized_width=width,
        convergence_probability=float(hf_convergence_probability),
    )


def adapter_disagreement(adapted_prediction: Array, shared_prediction: Array, epsilon: float = 1.0e-8) -> float:
    adapted = np.asarray(adapted_prediction, dtype=float)
    shared = np.asarray(shared_prediction, dtype=float)
    if adapted.shape != shared.shape or adapted.ndim != 1:
        raise ValueError("Adapted and shared predictions must be equal-shaped vectors.")
    denominator = np.linalg.norm(adapted) + epsilon
    return float(np.linalg.norm(adapted - shared) / denominator)
