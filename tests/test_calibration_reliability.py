from __future__ import annotations

import numpy as np

from domain_gated_mf.calibration import fit_marginal_calibration
from domain_gated_mf.reliability import (
    ApplicabilityDomainGate,
    GateEvidence,
    ReferenceLatentCloud,
    thresholds_from_calibration,
)
from domain_gated_mf.schema import GateThresholds


def test_calibration_is_componentwise_and_frozen():
    observed = np.asarray([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0], [3.0, 7.0]])
    predicted = observed - np.asarray([0.1, 0.2])
    scale = np.ones_like(observed) * np.asarray([0.1, 0.1])
    calibration = fit_marginal_calibration(observed, predicted, scale, nominal_coverage=0.75)
    assert np.allclose(calibration.quantile_multiplier, [1.0, 2.0])
    lower, upper = calibration.interval(predicted, scale)
    assert np.all(observed >= lower - 1.0e-7) and np.all(observed <= upper + 1.0e-7)
    assert calibration.quantile_multiplier.flags.writeable is False


def test_reference_cloud_uses_minimum_regularized_mahalanobis_distance():
    values = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    cloud = ReferenceLatentCloud.fit(values)
    assert cloud.distance(np.asarray([0.0, 0.0])) < 1.0e-5
    batch = cloud.distance(np.asarray([[0.0, 0.0], [2.0, 2.0]]))
    assert batch.shape == (2,) and batch[1] > batch[0]


def test_gate_is_conjunction_and_thresholds_follow_calibration_quantiles():
    thresholds = thresholds_from_calibration([1.0, 2.0, 3.0], [0.1, 0.2, 0.3], [0.5, 0.8, 0.9])
    assert thresholds.distance == 3.0
    assert thresholds.width == 0.3
    assert thresholds.convergence == 0.9
    gate = ApplicabilityDomainGate(GateThresholds(2.0, 0.2, 0.7))
    accepted = gate.decide(GateEvidence(1.0, 0.2, 0.7))
    rejected = gate.decide(GateEvidence(1.0, 0.3, 0.7))
    assert accepted.accepted and not rejected.accepted
    assert "calibrated interval width exceeds threshold" in rejected.reasons


def test_convergence_threshold_uses_true_calibration_labels_when_supplied():
    thresholds = thresholds_from_calibration(
        [1.0, 2.0],
        [0.1, 0.2],
        [0.2, 0.8],
        converged_labels=[False, True],
    )
    assert thresholds.convergence == 0.8
