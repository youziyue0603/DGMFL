from __future__ import annotations

from typing import Iterable

import numpy as np

from .schema import Array


def nondominated(points: Array) -> Array:
    """Return nondominated points for minimization objectives."""
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        return np.empty((0, values.shape[1] if values.ndim == 2 else 0), dtype=float)
    keep = np.ones(values.shape[0], dtype=bool)
    for i, point in enumerate(values):
        others = np.arange(values.shape[0]) != i
        keep[i] = not np.any(np.all(values[others] <= point, axis=1) & np.any(values[others] < point, axis=1))
    result = values[keep]
    if result.shape[0] > 1:
        _, unique_indices = np.unique(np.round(result, 12), axis=0, return_index=True)
        result = result[np.sort(unique_indices)]
    return result


def hypervolume(points: Array, reference_point: Array) -> float:
    """Exact dominated hypervolume for minimization points.

    The recursive sweep is practical for the small objective fronts used by
    the paper. Points outside the reference box cannot contribute.
    """
    values = nondominated(points)
    reference = np.asarray(reference_point, dtype=float)
    if reference.ndim != 1 or values.ndim != 2 or values.shape[1] != reference.size:
        raise ValueError("Point and reference dimensions do not match.")
    if values.shape[0] == 0:
        return 0.0
    values = values[np.all(values < reference, axis=1)]
    if values.shape[0] == 0:
        return 0.0
    if reference.size == 1:
        return float(reference[0] - np.min(values[:, 0]))
    coordinates = np.unique(np.concatenate((values[:, 0], np.asarray([reference[0]]))))
    coordinates.sort()
    total = 0.0
    for left, right in zip(coordinates[:-1], coordinates[1:]):
        if right <= left:
            continue
        active = values[values[:, 0] <= left, 1:]
        if active.shape[0]:
            total += float(right - left) * hypervolume(active, reference[1:])
    return float(max(total, 0.0))


def normalized_hypervolume(points: Array, reference_point: Array, reference_range: float | None = None) -> float:
    raw = hypervolume(points, reference_point)
    if reference_range is None:
        reference_range = float(np.prod(np.asarray(reference_point, dtype=float)))
    if reference_range <= 0.0:
        raise ValueError("Reference range must be positive.")
    return raw / reference_range


def igd_plus(approximation: Array, reference_front: Array) -> float:
    """Inverted generational distance plus from manuscript Eq. (metric)."""
    approx = np.asarray(approximation, dtype=float)
    reference = np.asarray(reference_front, dtype=float)
    if approx.ndim != 2 or reference.ndim != 2 or approx.shape[1] != reference.shape[1]:
        raise ValueError("Approximation and reference fronts must have equal objective dimensions.")
    if approx.shape[0] == 0 or reference.shape[0] == 0:
        return float("inf")
    distances = []
    for target in reference:
        delta = np.maximum(approx - target, 0.0)
        distances.append(np.min(np.linalg.norm(delta, axis=1)))
    return float(np.mean(distances))


def picp(observed: Array, lower: Array, upper: Array) -> float:
    observed_value = np.asarray(observed, dtype=float)
    lower_value = np.asarray(lower, dtype=float)
    upper_value = np.asarray(upper, dtype=float)
    if observed_value.shape != lower_value.shape or observed_value.shape != upper_value.shape:
        raise ValueError("Observed values and interval bounds must have equal shapes.")
    return float(np.mean((observed_value >= lower_value) & (observed_value <= upper_value)))


def negative_transfer_rate(transferred_error: Array, independent_error: Array) -> float:
    transferred = np.asarray(transferred_error, dtype=float)
    independent = np.asarray(independent_error, dtype=float)
    if transferred.shape != independent.shape:
        raise ValueError("Transferred and independent errors must have equal shapes.")
    if transferred.ndim != 1 or transferred.size == 0:
        raise ValueError("Errors must be a nonempty task-level vector.")
    return float(np.mean(transferred > independent))


def relative_field_error(predicted: Array, observed: Array, eps: float = 1.0e-12) -> float:
    prediction = np.asarray(predicted, dtype=float)
    target = np.asarray(observed, dtype=float)
    if prediction.shape != target.shape:
        raise ValueError("Predicted and observed fields must have equal shapes.")
    return float(np.linalg.norm(prediction - target) / (np.linalg.norm(target) + eps))


def hypervolume_improvement(points: Array, candidate: Array, reference_point: Array) -> float:
    before = hypervolume(points, reference_point)
    after = hypervolume(np.vstack((np.asarray(points, dtype=float), np.asarray(candidate, dtype=float))), reference_point)
    return float(after - before)


def paired_mean_delta(values_a: Iterable[float], values_b: Iterable[float]) -> float:
    first = np.asarray(list(values_a), dtype=float)
    second = np.asarray(list(values_b), dtype=float)
    if first.shape != second.shape or first.ndim != 1 or first.size == 0:
        raise ValueError("Paired vectors must have equal nonempty one-dimensional shapes.")
    return float(np.mean(first - second))

