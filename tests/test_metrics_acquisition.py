from __future__ import annotations

import numpy as np

from domain_gated_mf.acquisition import CandidatePrediction, residual_resampled_hv_gain
from domain_gated_mf.demo import make_demo_tasks
from domain_gated_mf.metrics import hypervolume, igd_plus, negative_transfer_rate, picp, relative_field_error
from domain_gated_mf.schema import AcquisitionConfig, Fidelity


def test_metrics_follow_minimization_definitions():
    points = np.asarray([[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]])
    assert hypervolume(points, np.asarray([1.0, 1.0])) == 0.37
    assert igd_plus(points, points) == 0.0
    assert picp(np.asarray([0.0, 1.0]), np.asarray([-0.1, 0.9]), np.asarray([0.1, 1.1])) == 1.0
    assert negative_transfer_rate([0.3, 0.7], [0.4, 0.6]) == 0.5
    assert relative_field_error([1.0, 2.0], [1.0, 2.0]) == 0.0


def test_lower_quartile_gain_is_deterministic_for_a_seed():
    task = make_demo_tasks()[0]
    existing = np.empty((0, 3))
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    gain_a = residual_resampled_hv_gain(task, existing, np.asarray([0.2, 0.2, 0.2, 0.0]), np.zeros(4), rng_a, AcquisitionConfig(resamples=64))
    gain_b = residual_resampled_hv_gain(task, existing, np.asarray([0.2, 0.2, 0.2, 0.0]), np.zeros(4), rng_b, AcquisitionConfig(resamples=64))
    assert gain_a == gain_b and gain_a > 0.0


def test_acquisition_uses_frozen_calibrated_scale_when_present():
    response = np.asarray([0.2, 0.2, 0.2, 0.0])
    prediction = CandidatePrediction(
        responses={level: response for level in Fidelity},
        hf_scale=np.ones(4),
        convergence_probability={level: 1.0 for level in Fidelity},
        feasibility_probability={level: 1.0 for level in Fidelity},
        adapted_hf_prediction=response,
        shared_hf_prediction=response,
        adapted_representation=np.zeros(2),
        calibrated_hf_scale=np.full(4, 0.25),
    )
    assert np.allclose(prediction.acquisition_scale, 0.25)
