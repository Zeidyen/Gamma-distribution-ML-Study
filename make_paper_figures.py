"""Figures for paper 1, in plain matplotlib, written to results/paper_figures/.

Fig 1  survival prediction error vs n, correctly specified heads (sim_recovery)
Fig 2  relative error of S at the true 1% time under misspecification (sim_misspec)
Fig 3  excess test log-loss of each head inside the network (sim_network)
Fig 4  integrated Brier score per test fold on the four datasets (realdata)

Never overwrites: if a figure file already exists it stops and says so.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
OUT = RES / "paper_figures"
OUT.mkdir(exist_ok=True)


def save(fig, name):
    path = OUT / name
    if path.exists():
        raise SystemExit(f"{path} exists; not overwriting. Move it or pick a new name.")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------- Fig 1
v = pd.read_csv(RES / "survival_summary.csv")
v = v[(v.fits == "all") & (v.cens == 0.3) & ~v.family.isin(["gb2_genf", "gb2_lmom"])]
labels = {"weibull": "Weibull", "burr12": "Burr XII", "er_burr12": "ER-BurrXII", "gb2": "GB2"}
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True)
for ax, (col, s) in zip(axes, [("rmse_S0.5", "0.5"), ("rmse_S0.1", "0.1"), ("rmse_S0.01", "0.01")]):
    for (fam, lab), g in v.groupby(["family", "setting"], sort=False):
        ax.plot(g.n, g[col], marker="o", label=f"{labels[fam]} {lab}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([250, 1000, 4000], ["250", "1000", "4000"])
    ax.minorticks_off()
    ax.set_xlabel("Sample size n")
    ax.set_ylabel("RMSE of fitted S(t)")
    ax.set_title(f"S(t) at the true S = {s} time")
axes[2].legend(fontsize=8)
save(fig, "fig1_survival_error_vs_n.png")

# ---------------------------------------------------------------- Fig 2
m = pd.read_csv(RES / "misspec_summary.csv")
m = m[(m.n == 1000) & (m.cens == "random30")]
truths = list(dict.fromkeys(m.truth))
models = [("weibull", "Weibull"), ("lognormal", "Log-normal"), ("burr12", "Burr XII"),
          ("er_burr12", "ER-BurrXII"), ("gb2_lmom", "GB2")]
fig, ax = plt.subplots(figsize=(12, 4.2))
w = 0.16
for j, (key, name) in enumerate(models):
    vals = [m[(m.truth == t) & (m.model == key)]["relerr_S0.01"].iloc[0] for t in truths]
    ax.bar(np.arange(len(truths)) + (j - 2) * w, vals, w, label=name)
ax.set_yscale("log")
ax.set_xticks(range(len(truths)), truths)
ax.set_xlabel("True distribution")
ax.set_ylabel("Relative RMSE of S at true 1% time")
ax.set_title("Tail prediction error by fitted model and true distribution (n = 1000, 30% censoring)")
ax.legend(fontsize=8, ncol=5)
save(fig, "fig2_misspecification_tail_error.png")

# ---------------------------------------------------------------- Fig 3
n = pd.read_csv(RES / "network_raw.csv")
n = n[n.n == 5000]
n["m"] = n.model + "/" + n.shapes
heads = [("weibull/global", "Weibull"), ("lognormal/global", "Log-normal"), ("burr12/x", "Burr XII (x)"),
         ("er_burr12/x", "ER-BurrXII (x)"), ("maptb12/x", "MAPT-Burr XII (x)"), ("gb2_lmom/x", "GB2 (x)"),
         ("mix_weibull/x", "Mixture Weibull"), ("mix_lognormal/x", "Mixture log-normal"),
         ("mix_burr12/x", "Mixture Burr XII")]
truths = [("weibull_aft", "Weibull AFT"), ("lognormal", "Log-normal"), ("gb2_mixtail", "GB2, tail varies by patient")]
fig, ax = plt.subplots(figsize=(12, 4.2))
w = 0.09
for j, (key, name) in enumerate(heads):
    vals = [n[(n.truth == t) & (n.m == key)].excess_logloss.median() for t, _ in truths]
    ax.bar(np.arange(len(truths)) + (j - 4) * w, vals, w, label=name)
ax.set_yscale("log")
ax.set_xticks(range(len(truths)), [lab for _, lab in truths])
ax.set_xlabel("True distribution")
ax.set_ylabel("Excess test log-loss (median of 20 runs)")
ax.set_title("Network heads with nonlinear covariate effects (n = 5000, 30% censoring)")
ax.legend(fontsize=7, ncol=3)
save(fig, "fig3_network_excess_logloss.png")

# ---------------------------------------------------------------- Fig 4
d = pd.read_csv(RES / "real_raw.csv")
show = [("mix_burr12/x", "Mix Burr XII"), ("mix_lognormal/x", "Mix log-normal"), ("gb2_lmom/x", "GB2 (x)"),
        ("burr12/x", "Burr XII (x)"), ("rsf", "RSF"), ("cox", "Cox"), ("deepsurv", "DeepSurv"),
        ("deephit", "DeepHit")]
fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
for ax, ds in zip(axes, ["metabric", "gbsg", "support", "flchain"]):
    data = [d[(d.dataset == ds) & (d.model == key)].ibs.to_numpy() for key, _ in show]
    ax.boxplot(data, tick_labels=[lab for _, lab in show])
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    ax.set_title(ds.upper())
    ax.set_ylabel("Integrated Brier score")
fig.suptitle("Integrated Brier score over 25 test folds (5 x 5-fold cross-validation)")
save(fig, "fig4_real_data_ibs.png")
