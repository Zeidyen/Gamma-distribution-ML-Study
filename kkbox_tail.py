"""How heavy is the KKBox tail? Marginal fits without covariates (Section 3.5, Fig. S2).

Fits Weibull, log-normal, Burr XII and GB2 (log-moment form) by censored
maximum likelihood to all 2.8 million KKBox spells, and compares them with the
Kaplan-Meier estimate on log-log axes, where a power-law tail is a straight
line. Durations are whole days, so the likelihood is evaluated once per
distinct (duration, event) pair and weighted by its count.

Writes results/kkbox_tail.csv (fits) and results/kkbox_km.csv, and the figure
results/paper_figures/figS2_kkbox_tail_v2.{pdf,png} (never overwritten).

Usage: .venv/bin/python kkbox_tail.py
"""

import numpy as np
import pandas as pd
import torch

import kkbox_benchmark as kb
from heads import DTYPE, HEADS
from sim_recovery import OUT, default_start, fit_multistart

FITS = (("weibull", "Weibull"), ("lognormal", "Log-normal"), ("burr12", "Burr XII"), ("gb2_lmom", "GB2"))


def kaplan_meier(t, e):
    times, idx = np.unique(t, return_inverse=True)
    deaths = np.bincount(idx, weights=e)
    at_risk = len(t) - np.concatenate([[0], np.cumsum(np.bincount(idx))[:-1]])
    return times, np.cumprod(1 - deaths / at_risk)


def main():
    df, _ = kb.load_kkbox()
    t, e = df.duration.to_numpy(), df.event.to_numpy()
    km_t, km_s = kaplan_meier(t, e)
    pd.DataFrame(dict(t=km_t, S=km_s)).to_csv(OUT / "kkbox_km.csv", index=False)

    u = pd.DataFrame(dict(t=t, e=e)).value_counts().reset_index(name="w")
    tt, ee, w = (torch.tensor(u[c].to_numpy(dtype=float), dtype=DTYPE) for c in ("t", "e", "w"))
    rows, curves = [], {}
    for name, label in FITS:
        head = HEADS[name]()

        def nll(theta):
            raw = theta.expand(len(tt), -1)
            ll = torch.where(ee.bool(), head.log_pdf(tt, raw), head.log_surv(tt, raw))
            return -(w * ll).sum() / w.sum()

        theta, loss, gnorm = fit_multistart(nll, default_start(head, t), np.random.default_rng(1))
        th = torch.tensor(theta, dtype=DTYPE)
        par = {k: float(v) for k, v in head.params(th).items()}
        grid = torch.tensor(km_t, dtype=DTYPE)
        curves[label] = np.exp(head.log_surv(grid, th.expand(len(grid), -1)).detach().numpy())
        row = dict(model=label, loglik_per_spell=-loss, aic=2 * len(t) * loss + 2 * head.n_params, grad_norm=gnorm,
                   S_759=float(curves[label][-1]), **par)
        if name == "burr12":
            row["tail_index_ck"] = par["c"] * par["k"]
        rows.append(row)
    fits = pd.DataFrame(rows)
    # straight-line slope of log S on log t over the last part of follow-up
    m = (km_t >= 100)
    fits.attrs["km_slope_100_759"] = np.polyfit(np.log(km_t[m]), np.log(km_s[m]), 1)[0]
    fits.to_csv(OUT / "kkbox_tail.csv", index=False)
    pd.set_option("display.width", 200)
    print(fits.to_string(index=False))
    print(f"KM: S(759) = {km_s[-1]:.4f}; log-log slope over 100-759 days = {fits.attrs['km_slope_100_759']:.3f}")
    print(f"median duration {np.median(t):.0f}, share of events within 31 days {(t[e == 1] <= 31).mean():.3f}")
    plot(km_t, km_s, curves)


def plot(km_t, km_s, curves):
    import make_paper_figures_v2 as fig  # paper style
    plt = fig.plt
    f, ax = plt.subplots(figsize=(fig.WIDTH * 0.62, 2.6))
    ax.plot(km_t, km_s, color=fig.INK, lw=1.6, label="Kaplan-Meier", zorder=5)
    for (label, s), col in zip(curves.items(), fig.SLOTS):
        ax.plot(km_t, s, color=col, lw=1.1, ls="--" if label == "GB2" else "-", label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.plain_log(ax.xaxis)
    fig.plain_log(ax.yaxis)
    ax.set_xlabel("Days since subscription start")
    ax.set_ylabel("Proportion not churned")
    ax.legend(loc="lower left")
    fig.save(f, "figS2_kkbox_tail_v2")


if __name__ == "__main__":
    main()
