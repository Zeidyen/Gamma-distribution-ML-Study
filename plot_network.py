"""Figure for sim_network.py: excess log-loss and tail calibration by head."""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sim_recovery import OUT  # noqa: E402

d = pd.read_csv(OUT / "network_raw.csv")
d["m"] = d.model.map({"weibull": "Weibull", "lognormal": "Log-normal", "burr12": "Burr XII",
                      "er_burr12": "ER-BurrXII", "maptb12": "MAPT-BurrXII", "gb2_lmom": "GB2", "mix_weibull": "Mix Weibull",
                      "mix_lognormal": "Mix log-normal", "mix_burr12": "Mix Burr XII"})
d["m"] = d["m"] + np.where(d.model.str.startswith("mix"), "", d.shapes.map({"global": "", "x": " (x)"}))
order = ["Weibull", "Weibull (x)", "Log-normal", "Log-normal (x)", "Burr XII", "Burr XII (x)",
         "ER-BurrXII", "ER-BurrXII (x)", "MAPT-BurrXII", "MAPT-BurrXII (x)", "GB2", "GB2 (x)", "Mix Weibull", "Mix log-normal", "Mix Burr XII"]
truths = [("weibull_aft", "Truth: Weibull AFT"), ("lognormal", "Truth: log-normal"),
          ("gb2_mixtail", "Truth: GB2, tail differs by patient")]
fig, axes = plt.subplots(2, 3, figsize=(19, 7.2))
for j, (tr, title) in enumerate(truths):
    g = d[(d.truth == tr) & (d.n == 5000)].groupby("m")
    kl = g.excess_logloss.median().reindex(order)
    cal = g["cal_S0.01"].mean().reindex(order)
    colors = [f"C{i // 2}" for i in range(12)] + ["C6", "C7", "C8"]
    ax = axes[0, j]
    ax.bar(range(len(order)), kl, color=colors)
    ax.set_yscale("log")
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(len(order)), order, rotation=60, ha="right", fontsize=8)
    ax = axes[1, j]
    ax.bar(range(len(order)), cal, color=colors)
    ax.axhline(1, color="k", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xticks(range(len(order)), order, rotation=60, ha="right", fontsize=8)
axes[0, 0].set_ylabel("Excess test log-loss\n(median over 20 runs; lower is better)")
axes[1, 0].set_ylabel("Calibration at the true 1% time\n(predicted S / 0.01; 1 is ideal)")
fig.suptitle("Heads inside an MLP, n = 5000, 30% censoring. '(x)' = shape parameters predicted per patient",
             fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "network_results.png", dpi=150)
