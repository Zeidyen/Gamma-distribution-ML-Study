"""Figure: error of fitted S(t) vs n for every head (results/survival_summary.csv).

Uses all fits with a finite likelihood, including those flagged as failed.
"""

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sim_recovery import OUT, SETTINGS  # noqa: E402

v = pd.read_csv(OUT / "survival_summary.csv")
v = v[(v.fits == "all") & (v.cens == 0.3)]
levels = [("rmse_S0.5", "S = 0.5 (median)"), ("rmse_S0.1", "S = 0.1"), ("rmse_S0.01", "S = 0.01 (tail)")]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True)
for ax, (col, title) in zip(axes, levels):
    for i, (fam, lab, _) in enumerate(SETTINGS):
        g = v[(v.family == fam) & (v.setting == lab)]
        ax.plot(g.n, g[col], marker="o", ms=3, lw=1.2, color=f"C{i % 10}",
                ls="-" if lab in ("W", "A") else "--", label=f"{fam} {lab}")
    ax.set(xscale="log", yscale="log", title=title, xlabel="n", xticks=[250, 1000, 4000],
           xticklabels=["250", "1000", "4000"])
    ax.minorticks_off()
axes[0].set_ylabel("RMSE of fitted S(t)")
axes[2].legend(fontsize=7, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
fig.suptitle("Survival prediction error, 30% censoring, all fits including failed ones: "
             "every head within about 2x of Weibull fitted to Weibull data", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "survival_error.png", dpi=150)
