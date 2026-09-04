from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import picp
from .schema import Array


@dataclass(frozen=True)
class FrozenMarginalCalibration:
    """Component-wise frozen calibration of the HF predictive scale."""

    quantile_multiplier: Array
    response_range: Array
    nominal_coverage: float = 0.90
    epsilon_sigma: float = 1.0e-8
    epsilon_width: float = 1.0e-8
    provenance: str = ""

    def __post_init__(self) -> None:
        q = np.asarray(self.quantile_multiplier, dtype=float)
        response_range = np.asarray(self.response_range, dtype=float)
        if q.ndim != 1 or response_range.shape != q.shape:
            raise ValueError("Calibration vectors must be one-dimensional and equal in shape.")
        if not np.all(np.isfinite(q)) or np.any(q < 0.0) or not np.all(np.isfinite(response_range)) or np.any(response_range < 0.0):
            raise ValueError("Calibration values must be finite and nonnegative.")
        if not 0.0 < self.nominal_coverage < 1.0:
            raise ValueError("Nominal coverage must lie between zero and one.")
        q.setflags(write=False)
        response_range.setflags(write=False)
        object.__setattr__(self, "quantile_multiplier", q)
        object.__setattr__(self, "response_range", response_range)

    @property
    def dimension(self) -> int:
        return int(self.quantile_multiplier.size)

    def interval(self, mean: Array, scale: Array) -> tuple[Array, Array]:
        mean_value = np.asarray(mean, dtype=float)
        scale_value = np.asarray(scale, dtype=float)
        if mean_value.shape != scale_value.shape or mean_value.shape[-1] != self.dimension:
            raise ValueError("Prediction and scale shapes do not match calibration.")
        width = self.quantile_multiplier * np.maximum(scale_value, 0.0)
        return mean_value - width, mean_value + width

    def normalized_half_width(self, scale: Array) -> float | Array:
        scale_value = np.asarray(scale, dtype=float)
        if scale_value.shape[-1] != self.dimension:
            raise ValueError("Scale dimension does not match calibration.")
        width = self.quantile_multiplier * np.maximum(scale_value, 0.0)
        return width / (self.response_range + self.epsilon_width)

    def scalar_width(self, scale: Array, component: int = 0) -> float:
        widths = np.asarray(self.normalized_half_width(np.asarray(scale, dtype=float)))
        return float(np.max(widths[..., component]))

    def coverage(self, observed: Array, mean: Array, scale: Array) -> float:
        lower, upper = self.interval(mean, scale)
        return picp(np.asarray(observed, dtype=float), lower, upper)


def fit_marginal_calibration(
    observed_hf: Array,
    predicted_hf: Array,
    uncalibrated_scale: Array,
    nominal_coverage: float = 0.90,
    provenance: str = "",
    epsilon_sigma: float = 1.0e-8,
    epsilon_width: float = 1.0e-8,
) -> FrozenMarginalCalibration:
    observed = np.asarray(observed_hf, dtype=float)
    predicted = np.asarray(predicted_hf, dtype=float)
    scale = np.asarray(uncalibrated_scale, dtype=float)
    if observed.ndim != 2 or predicted.shape != observed.shape or scale.shape != observed.shape:
        raise ValueError("Calibration arrays must be equal-shaped two-dimensional arrays.")
    if observed.shape[0] < 1 or not np.all(np.isfinite(observed)) or not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(scale)):
        raise ValueError("Calibration samples must be nonempty and finite.")
    if np.any(scale < 0.0):
        raise ValueError("Uncalibrated scale must be nonnegative.")
    alpha = 1.0 - nominal_coverage
    scores = np.abs(observed - predicted) / (scale + epsilon_sigma)
    q = np.quantile(scores, 1.0 - alpha, axis=0, method="higher")
    response_range = np.ptp(observed, axis=0)
    response_range[response_range < epsilon_width] = 1.0
    return FrozenMarginalCalibration(q, response_range, nominal_coverage, epsilon_sigma, epsilon_width, provenance)

