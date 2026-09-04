from __future__ import annotations

import numpy as np

from domain_gated_mf.acquisition import CandidatePrediction, RiskCostFidelityAllocator
from domain_gated_mf.demo import make_demo_tasks
from domain_gated_mf.interfaces import EvaluationResult
from domain_gated_mf.optimizer import DomainGatedOptimizer
from domain_gated_mf.reliability import GateDecision, GateEvidence
from domain_gated_mf.schema import AcquisitionConfig, Fidelity, OptimizationConfig


class OneCandidate:
    def __init__(self, design):
        self.design = np.asarray(design, dtype=float)

    def generate(self, task, working_designs, count, rng):
        return np.repeat(self.design[None, :], count, axis=0)


class CountingEvaluator:
    def __init__(self):
        self.calls = []

    def evaluate(self, task, design, fidelity):
        self.calls.append(fidelity)
        response = np.asarray([0.2, 0.3, 0.4, -0.5])
        return EvaluationResult(response, True, True, 1.0)


class StubAllocator(RiskCostFidelityAllocator):
    def __init__(self, config, gain_positive):
        super().__init__(config)
        self.gain_positive = gain_positive

    def conservative_hf_gain(self, *args, **kwargs):
        return 1.0 if self.gain_positive else 0.0


class StubRuntime:
    def __init__(self, task, accepted, gain_positive):
        self.task = task
        self.allocator = StubAllocator(AcquisitionConfig(resamples=8), gain_positive)
        self.accepted = accepted
        self.gain_positive = gain_positive
        self.repository = None

    def evidence(self, design):
        prediction = CandidatePrediction(
            responses={level: np.asarray([0.2, 0.3, 0.4, -0.5]) for level in Fidelity},
            hf_scale=np.zeros(4),
            convergence_probability={level: 1.0 for level in Fidelity},
            feasibility_probability={level: 1.0 for level in Fidelity},
            adapted_hf_prediction=np.asarray([0.2, 0.3, 0.4, -0.5]),
            shared_hf_prediction=np.asarray([0.2, 0.3, 0.4, -0.5]),
            adapted_representation=np.zeros(3),
        )
        if self.accepted:
            gate = GateDecision(True, GateEvidence(0.0, 0.0, 1.0), ())
        else:
            gate = GateDecision(False, GateEvidence(1.0, 1.0, 0.0), ("forced test rejection",))
        return type("Evidence", (), {"prediction": prediction, "gate": gate, "disagreement": 0.0})()

    def update_after_batch(self, records):
        return None


def test_failed_gate_escalates_when_conservative_gain_is_positive():
    task = make_demo_tasks()[0]
    runtime = StubRuntime(task, accepted=False, gain_positive=True)
    evaluator = CountingEvaluator()
    result = DomainGatedOptimizer(
        task, runtime, evaluator, OneCandidate(np.asarray([0.1, 0.1, 0.1, 0.1])),
        OptimizationConfig(hf_budget=1, candidate_batch_size=1, max_batches=1, seed=1),
    ).run()
    assert evaluator.calls == [Fidelity.H]
    assert result.hf_calls == 1
    assert result.events[0].selected_fidelity is Fidelity.H


def test_failed_gate_rejects_when_conservative_gain_is_not_positive():
    task = make_demo_tasks()[0]
    runtime = StubRuntime(task, accepted=False, gain_positive=False)
    evaluator = CountingEvaluator()
    result = DomainGatedOptimizer(
        task, runtime, evaluator, OneCandidate(np.asarray([0.1, 0.1, 0.1, 0.1])),
        OptimizationConfig(hf_budget=1, candidate_batch_size=1, max_batches=1, seed=1),
    ).run()
    assert evaluator.calls == []
    assert result.hf_calls == 0
    assert result.events[0].selected_fidelity is None


def test_passed_gate_uses_fidelity_allocation_without_spending_hf_budget_on_lf():
    task = make_demo_tasks()[0]
    runtime = StubRuntime(task, accepted=True, gain_positive=True)
    evaluator = CountingEvaluator()
    result = DomainGatedOptimizer(
        task, runtime, evaluator, OneCandidate(np.asarray([0.1, 0.1, 0.1, 0.1])),
        OptimizationConfig(hf_budget=1, candidate_batch_size=1, max_batches=1, seed=1),
    ).run()
    assert evaluator.calls == [Fidelity.L]
    assert result.hf_calls == 0
    assert result.events[0].gate_accepted
    assert result.events[0].selected_fidelity is Fidelity.L
