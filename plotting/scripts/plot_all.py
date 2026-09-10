from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MATERIALS = ["LCE", "SMP", "Hydrogel", "DEA", "MRC", "PRF"]
MATERIAL_LABELS = {
    "LCE": "LCE",
    "SMP": "SMP",
    "Hydrogel": "Hydrogel",
    "DEA": "DEA",
    "MRC": "Magnetic composite",
    "PRF": "Photoresponsive film",
}
METHODS = [
    "NSGA-II",
    "MOEA/D",
    "SMS-EGO",
    "ParEGO",
    "GP-BO",
    "SF-SAEA",
    "Physics-free",
    "MF-SAEA",
    "Proposed",
]
COLORS = {
    "NSGA-II": "#B9B9C8",
    "MOEA/D": "#A7ACC5",
    "SMS-EGO": "#929DBD",
    "ParEGO": "#7C8DB3",
    "GP-BO": "#677DA8",
    "SF-SAEA": "#536D9C",
    "Physics-free": "#897AAE",
    "MF-SAEA": "#4B5F88",
    "Proposed": "#C65B65",
}
curve_methods = ["ParEGO", "GP-BO", "Physics-free", "MF-SAEA", "Proposed"]
transfer_methods = [
    "From scratch",
    "Direct fine-tuning",
    "Shared backbone",
    "Proposed + gate",
]
shots = [0, 5, 10, 20]
ood_scenarios = ["In domain", "Geometry shift", "Stimulus shift", "Coupled shift"]
field_methods = ["DeepONet", "PI-DeepONet", "P2INN", "Proposed"]
ablation_order = [
    "Full method",
    "w/o physics condition",
    "w/o fidelity decomposition",
    "w/o material adapter",
    "w/o field operator",
    "w/o uncertainty calibration",
    "w/o latent-distance gate",
    "w/o forced HF escalation",
    "w/o cross-material pretrain",
]
ablation_specs = {name: None for name in ablation_order}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.0,
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "figure.facecolor": "white",
    }
)


def add_panel_label(ax, label):
    display_label = label if str(label).startswith("(") else f"({label})"
    ax.text(
        -0.10,
        1.04,
        display_label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
    )


def save_figure(fig, stem, width_mm=183, height_mm=118):
    fig.set_size_inches(width_mm / 25.4, height_mm / 25.4)
    for ax in fig.axes:
        ax.grid(False, which="both", axis="both")
    fig.savefig(FIG_DIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(
        FIG_DIR / f"{stem}.tiff",
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


benchmark = pd.read_csv(DATA_DIR / "main_benchmark.csv")
summary = pd.read_csv(DATA_DIR / "main_summary.csv")
method_summary = pd.read_csv(DATA_DIR / "method_summary.csv")
curve_df = pd.read_csv(DATA_DIR / "convergence_curves.csv")
transfer = pd.read_csv(DATA_DIR / "few_shot_transfer.csv")
ntr_df = pd.read_csv(DATA_DIR / "negative_transfer.csv")
calibration = pd.read_csv(DATA_DIR / "calibration.csv")
ood = pd.read_csv(DATA_DIR / "ood_gate.csv")
risk_df = pd.read_csv(DATA_DIR / "risk_coverage.csv")
field = pd.read_csv(DATA_DIR / "field_benchmark.csv")
ablation = pd.read_csv(DATA_DIR / "ablation.csv")
sensitivity = pd.read_csv(DATA_DIR / "sensitivity.csv")
fidelity_robustness = pd.read_csv(DATA_DIR / "fidelity_robustness.csv")
scalability = pd.read_csv(DATA_DIR / "scalability.csv")


# 8. Figures
# -----------------------------------------------------------------------------
sns.set_theme(style="white", context="paper", font_scale=0.78)

# Figure 1: main benchmark
fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(2, 2, width_ratios=[1.65, 1.0], height_ratios=[1.0, 1.0])
ax_a = fig.add_subplot(gs[:, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 1])

hv_matrix = summary.pivot(index="method", columns="material", values="hv_mean").loc[
    METHODS, MATERIALS
]
cmap = LinearSegmentedColormap.from_list(
    "hvmap", ["#F1F2F7", "#AAB7D8", "#536D9C", "#C65B65"]
)
sns.heatmap(
    hv_matrix,
    ax=ax_a,
    cmap=cmap,
    vmin=0.63,
    vmax=0.89,
    annot=True,
    fmt=".3f",
    annot_kws={"fontsize": 5.8},
    cbar_kws={"label": "Normalized HV"},
    linewidths=0.35,
    linecolor="white",
)
ax_a.set_xlabel("Material family")
ax_a.set_ylabel("")
ax_a.set_xticklabels([MATERIAL_LABELS[m] for m in MATERIALS], rotation=35, ha="right")
ax_a.set_yticklabels(METHODS, rotation=0, ha="right", fontsize=6.2)
ax_a.tick_params(axis="y", pad=2)
add_panel_label(ax_a, "a")

rank_plot = method_summary.sort_values("avg_rank", ascending=True)
ax_b.barh(
    rank_plot["method"],
    rank_plot["avg_rank"],
    color=[COLORS[m] for m in rank_plot["method"]],
    height=0.68,
)
ax_b.invert_yaxis()
ax_b.set_xlabel("Average rank (lower is better)")
ax_b.set_ylabel("")
ax_b.set_xlim(0, 9.2)
for y, v in enumerate(rank_plot["avg_rank"]):
    ax_b.text(v + 0.12, y, f"{v:.2f}", va="center", fontsize=5.8)
add_panel_label(ax_b, "b")

for _, row in method_summary.iterrows():
    method = row["method"]
    ax_c.scatter(
        row["calls_mean"],
        row["hv_mean"],
        s=30 + 115 * (row["feasible_mean"] - 0.75),
        color=COLORS[method],
        edgecolor="white",
        linewidth=0.45,
        zorder=3,
    )
    ax_c.text(
        row["calls_mean"] + 1.2,
        row["hv_mean"] + (0.003 if method != "Proposed" else 0.006),
        method,
        fontsize=5.3,
        color="#333333",
    )
ax_c.set_xlabel("HF calls to target HV")
ax_c.set_ylabel("Mean normalized HV")
ax_c.set_xlim(48, 123)
ax_c.set_ylim(0.63, 0.88)
add_panel_label(ax_c, "c")
save_figure(fig, "Figure_4", 183, 118)

# Figure 2: convergence
fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
for ax, material in zip(axes.flat, MATERIALS):
    subset = curve_df.query("material == @material")
    stats_curve = (
        subset.groupby(["method", "hf_budget"])["hv"].agg(["mean", "std"]).reset_index()
    )
    for method in curve_methods:
        dat = stats_curve.query("method == @method")
        ax.plot(
            dat["hf_budget"],
            dat["mean"],
            color=COLORS[method],
            lw=1.35,
            marker="o",
            ms=2.1,
            label=method,
        )
        ax.fill_between(
            dat["hf_budget"].to_numpy(),
            (dat["mean"] - dat["std"]).to_numpy(),
            (dat["mean"] + dat["std"]).to_numpy(),
            color=COLORS[method],
            alpha=0.11,
            linewidth=0,
        )
    ax.set_title(MATERIAL_LABELS[material], fontsize=7, fontweight="bold")
    ax.set_xlim(20, 120)
    ax.set_ylim(0.49, 0.91)
    ax.tick_params(labelsize=5.8)
for ax in axes[-1, :]:
    ax.set_xlabel("High-fidelity evaluations")
for ax in axes[:, 0]:
    ax.set_ylabel("Normalized HV")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    ncol=5,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    fontsize=6.0,
)
for label, ax in zip(list("abcdef"), axes.flat):
    add_panel_label(ax, label)
save_figure(fig, "Figure_5", 183, 125)

# Figure 3: transfer and negative transfer
fig, axes = plt.subplots(1, 3, gridspec_kw={"width_ratios": [1.25, 1.15, 0.85]})
fig.subplots_adjust(wspace=0.55)
transfer_colors = {
    "From scratch": "#B9B9C8",
    "Direct fine-tuning": "#8FA0C2",
    "Shared backbone": "#536D9C",
    "Proposed + gate": "#C65B65",
}
agg = (
    transfer.groupby(["method", "hf_shots"])["relative_error"]
    .agg(["mean", "sem"])
    .reset_index()
)
for method in transfer_methods:
    dat = agg.query("method == @method")
    ci = 1.96 * dat["sem"]
    axes[0].plot(
        dat["hf_shots"],
        dat["mean"],
        marker="o",
        ms=3,
        lw=1.5,
        color=transfer_colors[method],
        label=method,
    )
    axes[0].fill_between(
        dat["hf_shots"].to_numpy(),
        (dat["mean"] - ci).to_numpy(),
        (dat["mean"] + ci).to_numpy(),
        color=transfer_colors[method],
        alpha=0.13,
        linewidth=0,
    )
axes[0].set_xlabel("High-fidelity samples for adaptation")
axes[0].set_ylabel("Relative response error")
axes[0].set_xticks(shots)
axes[0].legend(fontsize=5.5)
add_panel_label(axes[0], "a")

shot20 = (
    transfer.query("hf_shots == 20")
    .groupby(["method", "material"])["relative_error"]
    .mean()
    .reset_index()
)
mat = shot20.pivot(index="method", columns="material", values="relative_error").loc[
    transfer_methods, MATERIALS
]
sns.heatmap(
    mat,
    ax=axes[1],
    cmap="Blues",
    vmin=0.08,
    vmax=0.24,
    annot=True,
    fmt=".3f",
    annot_kws={"fontsize": 5.6},
    cbar_kws={"label": "Relative error"},
    linewidths=0.3,
    linecolor="white",
)
axes[1].set_xlabel("Held-out material family")
axes[1].set_ylabel("")
axes[1].set_xticklabels(MATERIALS, rotation=35, ha="right")
add_panel_label(axes[1], "b")

ntr_summary = (
    ntr_df.groupby("method")["negative_transfer"]
    .mean()
    .reindex(["No adapter", "No gate", "Full method"])
)
axes[2].bar(
    ntr_summary.index,
    100 * ntr_summary.values,
    color=["#B9B9C8", "#8FA0C2", "#C65B65"],
    width=0.68,
)
axes[2].set_ylabel("Negative-transfer rate (%)")
axes[2].set_ylim(0, 75)
axes[2].tick_params(axis="x", rotation=30)
for i, v in enumerate(100 * ntr_summary.values):
    axes[2].text(i, v + 2.0, f"{v:.1f}", ha="center", fontsize=6)
add_panel_label(axes[2], "c")
save_figure(fig, "Figure_6", 183, 108)

# Figure 4: calibration and OOD gate
fig, axes = plt.subplots(2, 2)
cal_agg = (
    calibration.groupby(["method", "target"])["coverage"]
    .agg(["mean", "sem"])
    .reset_index()
)
for method, color, marker in [
    ("Raw ensemble", "#536D9C", "s"),
    ("Calibrated", "#C65B65", "o"),
]:
    dat = cal_agg.query("method == @method")
    axes[0, 0].errorbar(
        dat["target"],
        dat["mean"],
        yerr=1.96 * dat["sem"],
        color=color,
        marker=marker,
        lw=1.3,
        capsize=2,
        label=method,
    )
axes[0, 0].plot([0.78, 0.97], [0.78, 0.97], ls="--", color="#777777", lw=0.8)
axes[0, 0].set_xlabel("Nominal coverage")
axes[0, 0].set_ylabel("Empirical coverage")
axes[0, 0].set_xlim(0.78, 0.97)
axes[0, 0].set_ylim(0.64, 0.98)
axes[0, 0].legend(fontsize=5.8)
add_panel_label(axes[0, 0], "a")

ood_agg = (
    ood.groupby(["scenario", "method"])[
        ["selective_rmse", "accepted_fraction", "violation_rate"]
    ]
    .mean()
    .reset_index()
)
scenario_order = ood_scenarios
full = ood_agg.query("method == 'Full gate'").set_index("scenario").loc[scenario_order]
nogate = ood_agg.query("method == 'No gate'").set_index("scenario").loc[scenario_order]
x = np.arange(len(scenario_order))
w = 0.36
axes[0, 1].bar(
    x - w / 2, nogate["selective_rmse"], width=w, color="#8FA0C2", label="No gate"
)
axes[0, 1].bar(
    x + w / 2, full["selective_rmse"], width=w, color="#C65B65", label="Full gate"
)
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(["ID", "Geometry", "Stimulus", "Coupled"])
axes[0, 1].set_ylabel("Selective RMSE")
axes[0, 1].legend(fontsize=5.8)
add_panel_label(axes[0, 1], "b")

risk_agg = (
    risk_df.groupby(["scenario", "method", "coverage"])["risk"].mean().reset_index()
)
for method, ls in [("Variance gate", "--"), ("Full gate", "-")]:
    for scenario, color in zip(
        ood_scenarios, ["#B9B9C8", "#8FA0C2", "#536D9C", "#C65B65"]
    ):
        dat = risk_agg.query("method == @method and scenario == @scenario")
        axes[1, 0].plot(
            dat["coverage"],
            dat["risk"],
            color=color,
            ls=ls,
            lw=1.1,
            label=f"{scenario} — {method}" if method == "Full gate" else None,
        )
axes[1, 0].set_xlabel("Accepted fraction")
axes[1, 0].set_ylabel("Risk among accepted predictions")
axes[1, 0].set_xlim(0.30, 1.00)
axes[1, 0].legend(fontsize=5.0, ncol=2)
add_panel_label(axes[1, 0], "c")

axes[1, 1].bar(
    x,
    100 * full["accepted_fraction"],
    color=["#B9B9C8", "#8FA0C2", "#536D9C", "#C65B65"],
    width=0.68,
)
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(["ID", "Geometry", "Stimulus", "Coupled"])
axes[1, 1].set_ylabel("Accepted predictions (%)")
axes[1, 1].set_ylim(0, 105)
for i, v in enumerate(100 * full["accepted_fraction"]):
    axes[1, 1].text(i, v + 2.0, f"{v:.0f}", ha="center", fontsize=5.8)
add_panel_label(axes[1, 1], "d")
save_figure(fig, "Figure_7", 183, 130)

# Figure 5: ablation and field prediction
fig, axes = plt.subplots(1, 3, gridspec_kw={"width_ratios": [1.30, 1.10, 1.0]})
abl = (
    ablation.groupby("variant")
    .agg(
        hv=("hv", "mean"),
        hf_calls=("hf_calls", "mean"),
        picp90=("picp90", "mean"),
        ntr=("ntr", "mean"),
        violation=("violation_rate", "mean"),
    )
    .reset_index()
)
full_row = abl.query("variant == 'Full method'").iloc[0]
abl_order = list(ablation_specs.keys())
abl = abl.set_index("variant").loc[abl_order].reset_index()
delta = pd.DataFrame(
    {
        "HV": abl["hv"] - full_row["hv"],
        "HF calls": (full_row["hf_calls"] - abl["hf_calls"]) / 100.0,
        "PICP90": abl["picp90"] - full_row["picp90"],
        "Violation": full_row["violation"] - abl["violation"],
    }
)
sns.heatmap(
    delta.iloc[1:, :],
    ax=axes[0],
    cmap="vlag",
    center=0,
    vmin=-0.10,
    vmax=0.10,
    yticklabels=[
        "−physics",
        "−fidelity",
        "−adapter",
        "−field op.",
        "−calibration",
        "−domain gate",
        "−HF escalation",
        "−pretraining",
    ],
    annot=True,
    fmt="+.3f",
    annot_kws={"fontsize": 5.1},
    cbar=False,
    linewidths=0.3,
)
axes[0].set_xlabel("")
axes[0].set_ylabel("")
axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0, fontsize=6.0)
axes[0].tick_params(axis="x", labelsize=6.0)
add_panel_label(axes[0], "a")

field_summary = (
    field.groupby(["method", "material"])["relative_l2"].mean().reset_index()
)
field_mat = field_summary.pivot(
    index="method", columns="material", values="relative_l2"
).loc[field_methods, MATERIALS]
sns.heatmap(
    field_mat,
    ax=axes[1],
    cmap="Blues",
    vmin=0.05,
    vmax=0.13,
    annot=True,
    fmt=".3f",
    annot_kws={"fontsize": 5.4},
    cbar=False,
    linewidths=0.3,
)
axes[1].set_xlabel("Material family")
axes[1].set_ylabel("")
axes[1].set_xticklabels(MATERIALS, rotation=35, ha="right")
axes[1].set_yticklabels(field_methods, rotation=0, ha="right", fontsize=6.0)
axes[1].tick_params(axis="x", labelsize=6.0)
add_panel_label(axes[1], "b")

metric_means = (
    field.groupby("method")[["relative_l2", "peak_error", "physics_residual"]]
    .mean()
    .loc[field_methods]
)
metric_norm = metric_means / metric_means.max(axis=0)
x = np.arange(len(field_methods))
w = 0.23
for idx, (metric, color) in enumerate(
    zip(metric_norm.columns, ["#B9B9C8", "#8FA0C2", "#C65B65"])
):
    axes[2].bar(
        x + (idx - 1) * w,
        metric_norm[metric],
        width=w,
        color=color,
        label={
            "relative_l2": "$L_2$ error",
            "peak_error": "Peak error",
            "physics_residual": "Physics residual",
        }[metric],
    )
axes[2].set_xticks(x)
axes[2].set_xticklabels(field_methods, rotation=30, ha="right", fontsize=6.0)
axes[2].set_ylabel("Normalized error")
axes[2].set_ylim(0, 1.15)
axes[2].legend(fontsize=5.4)
add_panel_label(axes[2], "c")
save_figure(fig, "Figure_8", 183, 112)

# Figure 6: sensitivity, fidelity mismatch, and scalability
fig, axes = plt.subplots(2, 2)

physics = (
    sensitivity.query("analysis == 'physics_weight'")
    .groupby("setting")
    .agg(hv_mean=("hv", "mean"), hv_sd=("hv", "std"))
    .reset_index()
)
px = np.arange(len(physics))
axes[0, 0].errorbar(
    px,
    physics["hv_mean"],
    yerr=physics["hv_sd"],
    color=COLORS["Proposed"],
    marker="o",
    ms=3.4,
    lw=1.4,
    capsize=2.0,
)
axes[0, 0].scatter(
    [3],
    [physics.loc[3, "hv_mean"]],
    s=32,
    color=COLORS["Proposed"],
    edgecolor="white",
    linewidth=0.6,
    zorder=4,
)
axes[0, 0].set_xticks(px)
axes[0, 0].set_xticklabels(["0", "0.03", "0.1", "0.3", "1", "3"])
axes[0, 0].set_xlabel(r"Physics-loss weight $\lambda$")
axes[0, 0].set_ylabel("Normalized HV")
axes[0, 0].set_ylim(0.80, 0.87)
add_panel_label(axes[0, 0], "a")

gate = (
    sensitivity.query("analysis == 'gate_quantile'")
    .groupby("setting")
    .agg(accepted=("accepted_fraction", "mean"), violation=("violation_rate", "mean"))
    .reset_index()
)
axes[0, 1].plot(
    100 * gate["accepted"],
    100 * gate["violation"],
    color="#536D9C",
    marker="o",
    ms=3.4,
    lw=1.4,
)
for _, row in gate.iterrows():
    axes[0, 1].annotate(
        f"{row.setting:.3g}",
        (100 * row.accepted, 100 * row.violation),
        xytext=(3, 3),
        textcoords="offset points",
        fontsize=5.2,
    )
axes[0, 1].set_xlabel("Accepted predictions (%)")
axes[0, 1].set_ylabel("Constraint violation (%)")
axes[0, 1].set_xlim(62, 100)
axes[0, 1].set_ylim(1.5, 8.2)
add_panel_label(axes[0, 1], "b")

fid = (
    fidelity_robustness.groupby(["correlation", "method"])["hv"]
    .agg(["mean", "sem"])
    .reset_index()
)
for method in ["MF-SAEA", "Proposed"]:
    dat = fid.query("method == @method")
    ci = 1.96 * dat["sem"]
    axes[1, 0].plot(
        dat["correlation"],
        dat["mean"],
        marker="o",
        ms=3.4,
        lw=1.4,
        color=COLORS[method],
        label=method,
    )
    axes[1, 0].fill_between(
        dat["correlation"].to_numpy(),
        (dat["mean"] - ci).to_numpy(),
        (dat["mean"] + ci).to_numpy(),
        color=COLORS[method],
        alpha=0.13,
        linewidth=0,
    )
axes[1, 0].set_xlabel("Low/high-fidelity correlation")
axes[1, 0].set_ylabel("Normalized HV")
axes[1, 0].set_xticks([0.3, 0.5, 0.7, 0.9])
axes[1, 0].set_ylim(0.69, 0.88)
axes[1, 0].legend(fontsize=5.8)
add_panel_label(axes[1, 0], "c")

scale = (
    scalability.groupby("material_families")
    .agg(
        training=("training_minutes", "mean"), parameters=("parameters_million", "mean")
    )
    .reset_index()
)
axes[1, 1].plot(
    scale["material_families"],
    scale["training"] / scale["training"].iloc[0],
    marker="o",
    ms=3.4,
    lw=1.4,
    color="#536D9C",
    label="Training time",
)
axes[1, 1].plot(
    scale["material_families"],
    scale["parameters"] / scale["parameters"].iloc[0],
    marker="s",
    ms=3.1,
    lw=1.4,
    color=COLORS["Proposed"],
    label="Parameters",
)
axes[1, 1].set_xlabel("Number of material families")
axes[1, 1].set_ylabel("Relative compute (one family = 1)")
axes[1, 1].set_xticks(range(1, 7))
axes[1, 1].set_ylim(0.9, 3.0)
axes[1, 1].legend(fontsize=5.8)
add_panel_label(axes[1, 1], "d")

for ax in axes.flat:
    ax.tick_params(labelsize=5.9)
save_figure(fig, "Figure_9", 183, 126)
