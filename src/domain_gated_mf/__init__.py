"""Domain-gated multi-fidelity framework from the final 2.0 manuscript."""

from .acquisition import AllocationScore, CandidatePrediction, RiskCostFidelityAllocator
from .calibration import FrozenMarginalCalibration, fit_marginal_calibration
from .interfaces import EvaluationResult
from .metrics import hypervolume, igd_plus, negative_transfer_rate, picp, relative_field_error
from .model import DomainGatedSurrogate
from .optimizer import DomainGatedOptimizer, SurrogateRuntime
from .records import EvaluationRecord, FidelityRepository, ReportedParetoArchive, WorkingCandidatePool
from .reliability import ApplicabilityDomainGate, ReferenceLatentCloud
from .schema import Fidelity, GateThresholds, TaskSpec

__all__ = [
    "AllocationScore",
    "ApplicabilityDomainGate",
    "CandidatePrediction",
    "DomainGatedOptimizer",
    "DomainGatedSurrogate",
    "EvaluationRecord",
    "EvaluationResult",
    "Fidelity",
    "FidelityRepository",
    "FrozenMarginalCalibration",
    "GateThresholds",
    "ReferenceLatentCloud",
    "ReportedParetoArchive",
    "RiskCostFidelityAllocator",
    "SurrogateRuntime",
    "TaskSpec",
    "WorkingCandidatePool",
    "fit_marginal_calibration",
    "hypervolume",
    "igd_plus",
    "negative_transfer_rate",
    "picp",
    "relative_field_error",
]

