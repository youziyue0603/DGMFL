# Domain-Gated Multi-Fidelity Inverse Design

Code for reliability-aware multi-fidelity inverse design across smart-material tasks.

## Files

- `src/domain_gated_mf/interfaces.py`: task and fidelity interfaces.
- `src/domain_gated_mf/schema.py`: data structures and run settings.
- `src/domain_gated_mf/model.py`: shared encoder, task adapter, and response heads.
- `src/domain_gated_mf/training.py`: model training.
- `src/domain_gated_mf/calibration.py`: uncertainty calibration.
- `src/domain_gated_mf/reliability.py`: applicability-domain gate.
- `src/domain_gated_mf/acquisition.py`: fidelity allocation and exploration score.
- `src/domain_gated_mf/optimizer.py`: optimization loop and HF archive.
- `src/domain_gated_mf/metrics.py`: HV, IGD+, PICP, NTR, and field metrics.
- `examples/run_structural_demo.py`: quick run example.
- `scripts/audit_conformance.py`: parameter audit.
- `plotting/`: fixed figure data and final plotting code.
- `tests/`: unit tests.

## Run

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
python .\examples\run_structural_demo.py
```

To reproduce the quantitative figures:

```powershell
python -m pip install -r .\plotting\requirements.txt
python .\plotting\scripts\plot_all.py
```
