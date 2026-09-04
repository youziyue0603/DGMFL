from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from .acquisition import (
    CandidatePrediction,
    RiskCostFidelityAllocator,
    disagreement_from_predictions,
)
from .calibration import FrozenMarginalCalibration, fit_marginal_calibration
from .interfaces import CandidateGenerator, FidelityEvaluator
from .model import DomainGatedSurrogate
from .records import EvaluationRecord, FidelityRepository, ReportedParetoArchive, WorkingCandidatePool
from .reliability import (
    ApplicabilityDomainGate,
    GateDecision,
    ReferenceLatentCloud,
    make_gate_evidence,
    thresholds_from_calibration,
)
from .schema import AcquisitionConfig, Fidelity, GateThresholds, OptimizationConfig, TaskSpec


class UniformLocalCandidateGenerator:
    """Deterministic bounded candidate source used by the demonstration suite."""

    def __init__(self, local_probability: float = 0.75, local_scale: float = 0.12) -> None:
        if not 0.0 <= local_probability <= 1.0 or local_scale <= 0.0:
            raise ValueError("Invalid candidate-generator parameters.")
        self.local_probability = float(local_probability)
        self.local_scale = float(local_scale)

    def generate(self, task: TaskSpec, working_designs: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
        if count <= 0:
            return np.empty((0, task.design_dim), dtype=float)
        lower = np.asarray(task.design_lower, dtype=float)
        upper = np.asarray(task.design_upper, dtype=float)
        span = upper - lower
        pool = np.asarray(working_designs, dtype=float)
        output = np.empty((count, task.design_dim), dtype=float)
        for i in range(count):
            if pool.ndim == 2 and pool.shape[0] and rng.random() < self.local_probability:
                center = pool[int(rng.integers(pool.shape[0]))]
                output[i] = center + rng.normal(0.0, self.local_scale, task.design_dim) * span
            else:
                output[i] = lower + rng.random(task.design_dim) * span
        return np.clip(output, lower, upper)


@dataclass(frozen=True)
class RuntimeEvidence:
    prediction: CandidatePrediction
    gate: GateDecision
    disagreement: float


class SurrogateRuntime:
    """Frozen reliability state plus the model used by Algorithm 1."""

    def __init__(
        self,
        task: TaskSpec,
        model: DomainGatedSurrogate,
        acquisition_config: AcquisitionConfig | None = None,
        regularization: float = 1.0e-6,
        update_hook: Callable[[tuple[EvaluationRecord, ...]], None] | None = None,
    ) -> None:
        self.task = task
        self.model = model
        self.acquisition_config = acquisition_config or AcquisitionConfig()
        self.allocator = RiskCostFidelityAllocator(self.acquisition_config)
        self.regularization = regularization
        self.repository = FidelityRepository()
        self.calibration: FrozenMarginalCalibration | None = None
        self.reference_cloud: ReferenceLatentCloud | None = None
        self.gate: ApplicabilityDomainGate | None = None
        self.update_hook = update_hook

    def prepare(
        self,
        training_records: Iterable[EvaluationRecord],
        calibration_records: Iterable[EvaluationRecord],
        thresholds: GateThresholds | None = None,
    ) -> None:
        training = tuple(record for record in training_records if record.task == self.task.name)
        calibration = tuple(record for record in calibration_records if record.task == self.task.name)
        if not training or not calibration:
            raise ValueError("Training and calibration records are required for runtime preparation.")
        self.allocator.runtime_estimator.set_calibration_priors(calibration)
        training_descriptors = np.vstack([self.task.descriptor(record.design) for record in training])
        _, _, training_latents = self.model.shared_and_adapted_hf_numpy(training_descriptors, self.task.name)
        self.reference_cloud = ReferenceLatentCloud.fit(training_latents, self.regularization)

        observed = np.vstack([record.response for record in calibration])
        predicted = []
        scales = []
        distances = []
        widths = []
        convergence = []
        for record in calibration:
            prediction = self.predict(record.design)
            predicted.append(prediction.responses[Fidelity.H])
            scales.append(prediction.hf_scale)
            distances.append(float(self.reference_cloud.distance(prediction.adapted_representation)))
            convergence.append(prediction.convergence_probability[Fidelity.H])
        self.calibration = fit_marginal_calibration(
            observed,
            np.vstack(predicted),
            np.vstack(scales),
            nominal_coverage=1.0 - self.acquisition_config.alpha,
            provenance=f"{self.task.name}: frozen calibration split",
        )
        for scale in scales:
            widths.append(float(np.max(self.calibration.normalized_half_width(scale))))
        if thresholds is None:
            thresholds = thresholds_from_calibration(
                distances,
                widths,
                convergence,
                converged_labels=np.asarray([record.converged for record in calibration], dtype=bool),
            )
        self.gate = ApplicabilityDomainGate(thresholds)

    def predict(self, design: np.ndarray) -> CandidatePrediction:
        output = self.model.predict_numpy(self.task.descriptor(design), self.task.name)
        responses = {level: output.responses[level].detach().cpu().numpy().astype(float) for level in Fidelity}
        scale = output.hf_scale.detach().cpu().numpy().astype(float)
        convergence = {level: float(torch_sigmoid(output.convergence_logits[level]).item()) for level in Fidelity}
        feasibility = {level: float(torch_sigmoid(output.feasibility_logits[level]).item()) for level in Fidelity}
        adapted_hf, shared_hf, latent = self.model.shared_and_adapted_hf_numpy(
            self.task.descriptor(design), self.task.name
        )
        return CandidatePrediction(
            responses=responses,
            hf_scale=scale,
            convergence_probability=convergence,
            feasibility_probability=feasibility,
            adapted_hf_prediction=adapted_hf[0],
            shared_hf_prediction=shared_hf[0],
            adapted_representation=latent[0],
            calibrated_hf_scale=None
            if self.calibration is None
            else self.calibration.quantile_multiplier * scale,
        )

    def evidence(self, design: np.ndarray) -> RuntimeEvidence:
        if self.reference_cloud is None or self.calibration is None or self.gate is None:
            raise RuntimeError("Runtime must be prepared before candidates are scored.")
        prediction = self.predict(design)
        evidence = make_gate_evidence(
            self.reference_cloud,
            prediction.adapted_representation,
            prediction.hf_scale,
            self.calibration,
            prediction.convergence_probability[Fidelity.H],
        )
        disagreement = disagreement_from_predictions(
            prediction.adapted_hf_prediction,
            prediction.shared_hf_prediction,
            self.acquisition_config.eps_uncertainty,
        )
        return RuntimeEvidence(prediction, self.gate.decide(evidence), disagreement)

    def observe(self, record: EvaluationRecord) -> None:
        self.repository.add(record)

    def update_after_batch(self, records: tuple[EvaluationRecord, ...]) -> None:
        """Update the surrogate after a batch while keeping calibration frozen."""
        if self.update_hook is not None and records:
            self.update_hook(records)


def torch_sigmoid(value):
    # Kept as a local helper so runtime prediction accepts scalar torch tensors
    # without making the public data contract depend on torch.
    import torch

    return torch.sigmoid(value)


@dataclass(frozen=True)
class OptimizationEvent:
    batch: int
    design: np.ndarray
    gate_accepted: bool
    gate_reasons: tuple[str, ...]
    selected_fidelity: Fidelity | None
    evaluated: bool
    hf_call_index: int | None
    conservative_hf_gain: float | None
    allocation_scores: tuple[float, ...] = ()


@dataclass(frozen=True)
class OptimizationResult:
    task: str
    archive: ReportedParetoArchive
    repository: FidelityRepository
    working_pool: WorkingCandidatePool
    events: tuple[OptimizationEvent, ...]
    hf_calls: int
    batches: int


class DomainGatedOptimizer:
    """Algorithm 1, including its conditional escalation branch."""

    def __init__(
        self,
        task: TaskSpec,
        runtime: SurrogateRuntime,
        evaluator: FidelityEvaluator,
        generator: CandidateGenerator | None = None,
        config: OptimizationConfig | None = None,
    ) -> None:
        if runtime.task.name != task.name:
            raise ValueError("Runtime task and optimizer task must match.")
        self.task = task
        self.runtime = runtime
        self.evaluator = evaluator
        self.generator = generator or UniformLocalCandidateGenerator()
        self.config = config or OptimizationConfig()

    def run(self, initial_records: Iterable[EvaluationRecord] = ()) -> OptimizationResult:
        repository = FidelityRepository()
        working = WorkingCandidatePool()
        archive = ReportedParetoArchive(self.task)
        initial = tuple(initial_records)
        repository.extend(initial)
        working.extend(initial)
        for record in initial:
            if record.fidelity is Fidelity.H:
                archive.try_add(record)
        self.runtime.repository = repository
        events: list[OptimizationEvent] = []
        rng = np.random.default_rng(self.config.seed)
        hf_calls = 0
        batch_index = 0
        while hf_calls < self.config.hf_budget and batch_index < self.config.max_batches:
            batch_index += 1
            batch_records: list[EvaluationRecord] = []
            designs = self.generator.generate(self.task, working.designs(self.task.name), self.config.candidate_batch_size, rng)
            for design in designs:
                if hf_calls >= self.config.hf_budget:
                    break
                evidence = self.runtime.evidence(design)
                selected: Fidelity | None = None
                evaluated = False
                gain: float | None = None
                scores: tuple[float, ...] = ()
                if evidence.gate.accepted:
                    allocations = self.runtime.allocator.score(
                        self.task,
                        evidence.prediction,
                        archive.objectives(),
                        evidence.disagreement,
                        repository,
                        rng,
                    )
                    scores = tuple(item.score for item in allocations)
                    chosen = self.runtime.allocator.select(allocations)
                    selected = chosen.fidelity
                    if selected is Fidelity.H and hf_calls >= self.config.hf_budget:
                        selected = None
                    if selected is not None:
                        record = self._evaluate(design, selected)
                        self._record(repository, working, archive, record)
                        batch_records.append(record)
                        evaluated = True
                        if selected is Fidelity.H:
                            hf_calls += 1
                else:
                    gain = self.runtime.allocator.conservative_hf_gain(
                        self.task,
                        evidence.prediction,
                        archive.objectives(),
                        rng,
                    )
                    if gain > 0.0 and hf_calls < self.config.hf_budget:
                        selected = Fidelity.H
                        record = self._evaluate(design, Fidelity.H)
                        self._record(repository, working, archive, record)
                        batch_records.append(record)
                        evaluated = True
                        hf_calls += 1
                events.append(
                    OptimizationEvent(
                        batch=batch_index,
                        design=np.asarray(design, dtype=float).copy(),
                        gate_accepted=evidence.gate.accepted,
                        gate_reasons=evidence.gate.reasons,
                        selected_fidelity=selected,
                        evaluated=evaluated,
                        hf_call_index=hf_calls if evaluated and selected is Fidelity.H else None,
                        conservative_hf_gain=gain,
                        allocation_scores=scores,
                    )
                )
            self.runtime.update_after_batch(tuple(batch_records))
        return OptimizationResult(self.task.name, archive, repository, working, tuple(events), hf_calls, batch_index)

    def _evaluate(self, design: np.ndarray, fidelity: Fidelity) -> EvaluationRecord:
        result = self.evaluator.evaluate(self.task, design, fidelity)
        return EvaluationRecord.from_result(self.task, design, fidelity, result)

    @staticmethod
    def _record(
        repository: FidelityRepository,
        working: WorkingCandidatePool,
        archive: ReportedParetoArchive,
        record: EvaluationRecord,
    ) -> None:
        repository.add(record)
        working.add(record)
        if record.fidelity is Fidelity.H:
            archive.try_add(record)
