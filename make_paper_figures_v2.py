"""Publication versions of the paper figures (v2), written to results/paper_figures/.

Same data as make_paper_figures.py. Drawn at the journal text width (34 pc =
5.67 in) in Computer Modern so that figure text matches the paper at 100%, and
saved as both PDF (vector, for LaTeX) and 300-dpi PNG (for Word).

Fig 1  survival prediction error vs n, correctly specified heads
Fig 2  heatmap: relative error of S at the true 1% time, fitted model x truth
Fig 3  dot plot: excess test log-loss of each network head, one panel per truth
Fig 4  integrated Brier score over 25 test folds, four datasets
Fig S1 parameter recovery RMSE vs n (supplement)

Colours: categorical slots checked with the dataviz validator (all pairs pass
CVD and normal-vision separation); line style and markers carry identity too.
Never overwrites an existing file.

Usage: .venv/bin/python make_paper_figures_v2.py
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
OUT = RES / "paper_figures"
OUT.mkdir(exist_ok=True)

WIDTH = 5.67  # inches, interact text width (34 pc)
INK, INK2, GRID = "#1f1f1e", "#52514e", "#e4e3df"
# validated categorical slots (blue, aqua, amber, green): all-pairs CVD dE >= 9.1
SLOTS = ["#2a78d6", "#1baf7a", "#eda100", "#008300"]
BLUE, GREY = "#2a78d6", "#9a9893"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "legend.frameon": False,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        path = OUT / f"{name}.{ext}"
        if path.exists():
            raise SystemExit(f"{path} exists; not overwriting. Move it or pick a new name.")
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
        print("wrote", OUT / f"{name}.{ext}")
    plt.close(fig)


def plain_log(axis):
    """Log axis labelled 0.01, 0.02, 0.05, 0.1, ... in plain decimals, no minor labels."""
    axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def log_n_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks([250, 1000, 4000], ["250", "1,000", "4,000"])
    ax.minorticks_off()
    ax.set_xlim(200, 5000)


# ---------------------------------------------------------------- Fig 1
def fig1():
    v = pd.read_csv(RES / "survival_summary.csv")
    v = v[(v.fits == "all") & (v.cens == 0.3) & ~v.family.isin(["gb2_genf", "gb2_lmom"])]
    fam = {"weibull": ("Weibull", SLOTS[0], "o"), "burr12": ("Burr XII", SLOTS[1], "s"),
           "er_burr12": ("ER-BurrXII", SLOTS[2], "^"), "gb2": ("GB2", SLOTS[3], "D")}
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.25))
    for ax, (col, s) in zip(axes, [("rmse_S0.5", "0.5"), ("rmse_S0.1", "0.1"), ("rmse_S0.01", "0.01")]):
        for (f, lab), g in v.groupby(["family", "setting"], sort=False):
            name, c, mk = fam[f]
            ax.plot(g.n, g[col], color=c, marker=mk, ms=4, lw=1.4, ls="--" if lab == "B" else "-",
                    mfc=c if lab != "B" else "white", mew=1.0)
        log_n_axis(ax)
        ax.set_yscale("log")
        plain_log(ax.yaxis)
        ax.set_title(f"At the true $S = {s}$ time")
        ax.set_xlabel("Sample size $n$")
    axes[0].set_ylabel("RMSE of fitted $S(t)$")
    handles = [Line2D([], [], color=c, marker=mk, ms=4, lw=1.4, label=name) for name, c, mk in fam.values()]
    handles += [Line2D([], [], color=INK2, lw=1.2, ls="-", marker="o", ms=3.5, label="Setting A (W)"),
                Line2D([], [], color=INK2, lw=1.2, ls="--", marker="o", ms=3.5, mfc="white", label="Setting B")]
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.02),
               handlelength=2.2, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=1.2)
    save(fig, "fig1_survival_error_vs_n_v2")


# ---------------------------------------------------------------- Fig 2
def fig2():
    m = pd.read_csv(RES / "misspec_summary.csv")
    m = m[(m.n == 1000) & (m.cens == "random30")]
    truths = [("weibull W", "Weibull", "weibull"), ("lognormal LN", "Log-normal", "lognormal"),
              ("burr12 A", "Burr XII A", "burr12"), ("burr12 heavy", "Burr XII, heavy tail", "burr12"),
              ("er_burr12 A", "ER-BurrXII A", "er_burr12"), ("er_burr12 B", "ER-BurrXII B", "er_burr12"),
              ("gb2 A", "GB2 A", "gb2_lmom"), ("gb2 B", "GB2 B", "gb2_lmom")]
    models = [("weibull", "Weibull"), ("lognormal", "Log-normal"), ("burr12", "Burr XII"),
              ("er_burr12", "ER-BurrXII"), ("gb2_lmom", "GB2")]
    Z = np.array([[m[(m.truth == t) & (m.model == k)]["relerr_S0.01"].iloc[0] for k, _ in models]
                  for t, _, _ in truths])
    fig, ax = plt.subplots(figsize=(WIDTH * 0.8, 3.0))
    norm = LogNorm(vmin=0.1, vmax=10)
    im = ax.imshow(Z, cmap=BLUES, norm=norm, aspect="auto")
    ax.grid(False)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if Z[i, j] > 0.9 else INK)
    for i, (_, _, own) in enumerate(truths):
        j = [k for k, _ in models].index(own)
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.3))
    ax.set_xticks(range(len(models)), [n for _, n in models])
    ax.set_yticks(range(len(truths)), [n for _, n, _ in truths])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Fitted model")
    ax.set_ylabel("True distribution")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("Relative error of $S$ at the true 1% time")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2)
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    cb.ax.yaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout()
    save(fig, "fig2_misspecification_tail_error_v2")


# ---------------------------------------------------------------- Fig 3
def fig3():
    n = pd.read_csv(RES / "network_raw.csv")
    n = n[n.n == 5000]
    n["m"] = n.model + "/" + n.shapes
    heads = [("weibull/global", "Weibull"), ("lognormal/global", "Log-normal"), ("burr12/x", "Burr XII (x)"),
             ("er_burr12/x", "ER-BurrXII (x)"), ("maptb12/x", "MAPT-Burr XII (x)"), ("gb2_lmom/x", "GB2 (x)"),
             ("mix_weibull/x", "Mixture, Weibull"), ("mix_lognormal/x", "Mixture, log-normal"),
             ("mix_burr12/x", "Mixture, Burr XII")]
    truths = [("weibull_aft", "Truth: Weibull AFT"), ("lognormal", "Truth: log-normal"),
              ("gb2_mixtail", "Truth: GB2, tail varies")]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.6), sharey=True)
    y = np.arange(len(heads))[::-1]
    for ax, (t, title) in zip(axes, truths):
        g = n[n.truth == t]
        med = np.array([g[g.m == k].excess_logloss.median() for k, _ in heads])
        lo = np.array([g[g.m == k].excess_logloss.quantile(0.25) for k, _ in heads])
        hi = np.array([g[g.m == k].excess_logloss.quantile(0.75) for k, _ in heads])
        ax.hlines(y, lo, hi, color=BLUE, lw=1.6, alpha=0.45)
        mix = np.array([k.startswith("mix") for k, _ in heads])
        ax.plot(med[~mix], y[~mix], "o", color=BLUE, ms=4.5, mec="white", mew=0.6)
        ax.plot(med[mix], y[mix], "s", color=BLUE, ms=4.5, mec="white", mew=0.6)
        ax.set_xscale("log")
        plain_log(ax.xaxis)
        subs = {"weibull_aft": (1.0, 3.0), "lognormal": (1.0, 2.0, 5.0), "gb2_mixtail": (1.0,)}[t]
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=subs))
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.axhline(2.5, color=GRID, lw=0.8)
    axes[0].set_yticks(y, [lab for _, lab in heads])
    axes[0].tick_params(axis="y", length=0)
    axes[1].set_xlabel("Excess test log-loss (median and interquartile range of 20 runs; log scale)")
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig3_network_excess_logloss_v2")


# ---------------------------------------------------------------- Fig 4
def fig4():
    d = pd.read_csv(RES / "real_raw.csv")
    show = [("mix_burr12/x", "Mixture, Burr XII", True), ("mix_lognormal/x", "Mixture, log-normal", True),
            ("gb2_lmom/x", "GB2 (x)", True), ("burr12/x", "Burr XII (x)", True),
            ("rsf", "Random survival forest", False), ("cox", "Cox (linear)", False),
            ("deepsurv", "DeepSurv", False), ("deephit", "DeepHit", False)]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.3), sharey=True)
    rng = np.random.default_rng(0)
    y = np.arange(len(show))[::-1]
    for ax, (ds, title) in zip(axes.flat, [("metabric", "METABRIC"), ("gbsg", "GBSG"),
                                           ("support", "SUPPORT"), ("flchain", "FLCHAIN")]):
        for yi, (key, _, ours) in zip(y, show):
            v = 1000 * d[(d.dataset == ds) & (d.model == key)].ibs.to_numpy()
            c = BLUE if ours else GREY
            ax.boxplot(v, positions=[yi], vert=False, widths=0.55, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor=c, alpha=0.25, edgecolor=c, lw=0.8),
                       medianprops=dict(color=c, lw=1.4), whiskerprops=dict(color=c, lw=0.8),
                       capprops=dict(color=c, lw=0.8))
            ax.plot(v, yi + rng.uniform(-0.17, 0.17, len(v)), "o", color=c, ms=1.8, alpha=0.7, mew=0)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.axhline(3.5, color=GRID, lw=0.8)
    for ax in axes[:, 0]:
        ax.set_yticks(y, [lab for _, lab, _ in show])
        ax.tick_params(axis="y", length=0)
    fig.supxlabel(r"Integrated Brier score $\times$ 1000 (lower is better)", fontsize=8, y=0.065)
    fig.legend(handles=[Patch(fc=BLUE, alpha=0.4, ec=BLUE, label="Likelihood heads (this paper)"),
                        Patch(fc=GREY, alpha=0.4, ec=GREY, label="Comparators")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1), h_pad=1.2, w_pad=1.0)
    save(fig, "fig4_real_data_ibs_v2")


# ---------------------------------------------------------------- Fig S1
def figS1():
    r = pd.read_csv(RES / "recovery_summary.csv")
    names = {"weibull": "Weibull", "burr12": "Burr XII", "er_burr12": "ER-BurrXII", "gb2": "GB2 $(a,c,p,q)$",
             "gb2_genf": "GB2, generalized F", "gb2_lmom": "GB2, log-moment"}
    pretty = {"scale": "scale", "shape": "shape", "mu": r"$\mu$", "sigma": r"$\sigma$"}
    markers = ["o", "s", "^", "D"]
    panels = list(r.groupby(["family", "setting"], sort=False))
    fig, axes = plt.subplots(3, 4, figsize=(6.3, 5.6), sharex=True, sharey=True)
    for ax, ((fam, st), g) in zip(axes.flat, panels):
        params = list(dict.fromkeys(g.param))
        for i, p in enumerate(params):
            for cens, ls, mfc in ((0.0, "-", None), (0.3, "--", "white")):
                h = g[(g.param == p) & (g.cens == cens)]
                ax.plot(h.n, h.rmse_raw, color=SLOTS[i], marker=markers[i], ms=3.2, lw=1.2, ls=ls,
                        mfc=mfc or SLOTS[i], mew=0.9, label=pretty.get(p, f"${p}$") if cens == 0 else None)
        nn = np.array([250, 4000])
        ref = g[(g.n == 250) & (g.cens == 0.0)].rmse_raw.max() * np.sqrt(250 / nn)
        ax.plot(nn, ref, ":", color=INK2, lw=0.9)
        log_n_axis(ax)
        ax.set_yscale("log")
        plain_log(ax.yaxis)
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.set_ylim(0.006, 60)
        ax.set_title(f"{names[fam]} {st}", fontsize=8)
        ax.legend(loc="upper right", ncol=2, columnspacing=0.8, fontsize=6.5, handlelength=1.6, borderaxespad=0.2, labelspacing=0.2)
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(handles=[Line2D([], [], color=INK2, lw=1.2, ls="-", label="No censoring"),
                                  Line2D([], [], color=INK2, lw=1.2, ls="--", label="30% censoring"),
                                  Line2D([], [], color=INK2, lw=0.9, ls=":", label=r"Slope $n^{-1/2}$")],
                         loc="center", fontsize=7.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("RMSE (log parameter)")
    for ax in axes[-1]:
        ax.set_xlabel("Sample size $n$")
    axes[1, 3].xaxis.set_tick_params(labelbottom=True)
    axes[1, 3].set_xlabel("Sample size $n$")
    fig.tight_layout(h_pad=0.8, w_pad=0.5)
    save(fig, "figS1_recovery_rmse_v2")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); figS1()
