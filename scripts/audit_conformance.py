from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from domain_gated_mf.schema import FIDELITIES, TASK_FAMILIES, AcquisitionConfig, OptimizationConfig  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    contract_path = ROOT / "configs" / "paper_contract.yaml"
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    manuscript = (ROOT / contract["manuscript"]["source"]).resolve()
    if not manuscript.exists():
        raise SystemExit(f"Missing authoritative manuscript: {manuscript}")
    expected_hash = contract["manuscript"]["sha256"]
    actual_hash = sha256(manuscript)
    if actual_hash != expected_hash:
        raise SystemExit(f"Manuscript hash mismatch: expected {expected_hash}, got {actual_hash}")

    text = manuscript.read_text(encoding="utf-8")
    required_labels = ("eq:representation", "eq:field-operator", "eq:residual", "eq:gate", "eq:acquisition", "alg:domain-gated")
    missing_labels = [label for label in required_labels if f"label{{{label}}}" not in text]
    if missing_labels:
        raise SystemExit(f"Missing manuscript labels: {', '.join(missing_labels)}")

    protocol = contract["locked_protocol"]
    checks = {
        "task_families": tuple(protocol["task_families"]) == TASK_FAMILIES,
        "fidelity_levels": tuple(protocol["fidelity_levels"]) == tuple(level.value for level in FIDELITIES),
        "runs_per_task": protocol["runs_per_task"] == 15,
        "online_hf_budget": protocol["online_hf_budget"] == OptimizationConfig().hf_budget,
        "acquisition_resamples": protocol["acquisition_resamples"] == AcquisitionConfig().resamples,
        "six_task_mentions": all(re.search(rf"\b{name}\b", text) for name in TASK_FAMILIES),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(f"Conformance checks failed: {', '.join(failed)}")
    print("2.0 paper-to-code conformance audit: PASS")
    print(f"manuscript_sha256={actual_hash}")
    print(f"tasks={','.join(TASK_FAMILIES)}")
    print(f"fidelities={','.join(level.value for level in FIDELITIES)}")
    print(f"online_hf_budget={OptimizationConfig().hf_budget}")
    print(f"acquisition_resamples={AcquisitionConfig().resamples}")


if __name__ == "__main__":
    main()
