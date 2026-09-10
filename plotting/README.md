# Figure Reproduction

This directory contains the fixed source data and plotting code for the final
quantitative figures. The plotting script reads the CSV files without changing
their values.

## Files

- `data/main_summary.csv` and `data/method_summary.csv`:
  six-family benchmark results.
- `data/convergence_curves.csv`: convergence records.
- `data/few_shot_transfer.csv` and `data/negative_transfer.csv`: cross-material adaptation records.
- `data/calibration.csv`, `data/ood_gate.csv`, and `data/risk_coverage.csv`:
  calibration and applicability-domain results.
- `data/ablation.csv` and `data/field_benchmark.csv`: component and response-field results.
- `data/sensitivity.csv`, `data/fidelity_robustness.csv`, and `data/scalability.csv`:
  sensitivity and robustness records.
- `scripts/plot_all.py`: reads the fixed data and exports Figures 4-9.

## Run

```bash
python plotting/scripts/plot_all.py
```

Figures are written to `plotting/figures` as PDF, SVG, PNG, and TIFF files.
