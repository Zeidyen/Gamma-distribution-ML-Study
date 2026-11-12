"""Misspecification study: which head predicts survival best when the truth differs?

Every dataset is fitted with every model (paired design). Each true
distribution is crossed with each fitted model, sample size and censoring
scheme:

  random30  independent exponential censoring, 30% censored
  admin15   follow-up ends at the true time where 15% are still alive, so the
            S = 0.1, 0.05, 0.01 times lie beyond the data (tail extrapolation)

Metrics per fit
  S_hat at the true times where S = 0.5, 0.25, 0.1, 0.05, 0.01
  excess test log-loss: mean[log f_true - log f_hat] on 20 000 uncensored
  draws from the truth (a Monte Carlo estimate of KL(truth || fit))
  AIC, so an "AIC-selected" model can be scored as well.

Usage:  .venv/bin/python sim_misspec.py [--reps 100] [--workers 15] [--quick]
"""

import argparse
import itertools
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from scipy import optimize

from heads import DTYPE, HEADS
from sim_recovery import OUT, _seed, default_start, fit_multistart, GRAD_TOL

TRUTHS = [
    ("weibull", "W", dict(scale=1.0, shape=1.5)),
    ("lognormal", "LN", dict(mu=0.0, sigma=1.0)),
    ("burr12", "A", dict(a=1.0, c=3.0, k=1.0)),
    ("burr12", "heavy", dict(a=1.0, c=1.5, k=1.0)),  # tail index ck = 1.5
    ("er_burr12", "A", dict(a=1.0, c=3.0, k=1.5)),
    ("er_burr12", "B", dict(a=5.0, c=10.0, k=0.5)),
    ("gb2", "A", dict(a=1.0, c=2.0, p=1.5, q=1.5)),
    ("gb2", "B", dict(a=1.0, c=4.0, p=0.5, q=0.75)),
]
MODELS = ("weibull", "lognormal", "burr12", "er_burr12", "gb2_lmom")
SIZES = (250, 1000, 4000)
CENSORING = ("random30", "admin15")
S_LEVELS = (0.5, 0.25, 0.1, 0.05, 0.01)
N_TEST = 20_000

_TEST_CACHE = {}


def truth_raw(family, true):
    return torch.tensor(HEADS[family]().raw_from(**true), dtype=DTYPE)


def true_time(family, true, s):
    """Time at which the true S equals s."""
    head, raw = HEADS[family](), truth_raw(family, true)

    def gap(log_t):
        return float(head.log_surv(torch.tensor([np.exp(log_t)], dtype=DTYPE), raw)[0]) - np.log(s)

    return float(np.exp(optimize.brentq(gap, -60, 60)))


def censoring(family, true, scheme):
    """('rate', r) for exponential censoring or ('admin', tau)."""
    if scheme == "admin15":
        return ("admin", true_time(family, true, 0.15))
    t = HEADS[family]().sample(np.random.default_rng(0), 200_000, **true)
    return ("rate", optimize.brentq(lambda r: np.mean(-np.expm1(-r * t)) - 0.3, 1e-8, 1e6))


def test_set(family, label, true):
    """Fixed uncensored test sample and its true log density, cached per worker."""
    key = (family, label)
    if key not in _TEST_CACHE:
        head = HEADS[family]()
        t = torch.tensor(head.sample(np.random.default_rng([_seed(family, label), 7]), N_TEST, **true),
                         dtype=DTYPE)
        _TEST_CACHE[key] = (t, head.log_pdf(t, truth_raw(family, true)))
    return _TEST_CACHE[key]


def run_dataset(task):
    family, label, true, n, scheme, cens, t_s, rep = task
    torch.set_num_threads(1)
    rng = np.random.default_rng([_seed(family, label), n, CENSORING.index(scheme), rep, 11])
    t = HEADS[family]().sample(rng, n, **true)
    kind, value = cens
    c = rng.exponential(1 / value, size=n) if kind == "rate" else np.full(n, value)
    e = (t <= c).astype(float)
    t = np.minimum(t, c)
    tt, ee = torch.tensor(t, dtype=DTYPE), torch.tensor(e, dtype=DTYPE)
    t_test, logf_true = test_set(family, label, true)

    rows = []
    for model in MODELS:
        head = HEADS[model]()
        try:
            def nll(theta):
                return head.nll(theta.expand(n, -1), tt, ee)

            t0 = time.time()
            theta, loss, gnorm = fit_multistart(nll, default_start(head, t), rng)
            th = torch.tensor(theta, dtype=DTYPE)
            s_hat = torch.exp(head.log_surv(torch.tensor(t_s, dtype=DTYPE), th)).numpy()
            with torch.no_grad():
                kl = float((logf_true - head.log_pdf(t_test, th.expand(N_TEST, -1))).mean())
            err = None
        except Exception as exc:  # noqa: BLE001 - recorded, counted as a failed fit
            loss, gnorm, s_hat, kl, err = np.inf, np.inf, np.full(len(S_LEVELS), np.nan), np.nan, repr(exc)
            t0 = time.time()
        row = dict(truth=f"{family} {label}", n=n, cens=scheme, cens_obs=1 - e.mean(), rep=rep, model=model,
                   k=head.n_params, loss=loss, grad_norm=gnorm, converged=bool(np.isfinite(loss) and gnorm < GRAD_TOL),
                   aic=2 * n * loss + 2 * head.n_params, excess_logloss=kl, seconds=time.time() - t0, error=err)
        for s, v in zip(S_LEVELS, s_hat):
            row[f"S_hat_{s}"] = v
        rows.append(row)
    return rows


def add_aic_choice(df):
    """Append a pseudo-model 'aic' = the model with the lowest AIC on each dataset."""
    key = ["truth", "n", "cens", "rep"]
    ok = df[np.isfinite(df.loss)]
    best = ok.loc[ok.groupby(key).aic.idxmin()].copy()
    best["chosen"] = best.model
    best["model"] = "aic"
    return pd.concat([df, best], ignore_index=True)


def summarise(df):
    rows = []
    for (truth, n, cens, model), g in df.groupby(["truth", "n", "cens", "model"], sort=False):
        g = g[np.isfinite(g.loss)]
        row = dict(truth=truth, n=n, cens=cens, model=model, fits=len(g),
                   fail_pct=100 * (1 - g.converged.mean()) if model != "aic" else np.nan,
                   excess_logloss=g.excess_logloss.mean())
        for s in S_LEVELS:
            row[f"rmse_S{s}"] = np.sqrt(np.mean((g[f"S_hat_{s}"] - s) ** 2))
            row[f"relerr_S{s}"] = np.sqrt(np.mean((g[f"S_hat_{s}"] / s - 1) ** 2))
        if model == "aic":
            row["chosen"] = "; ".join(f"{k} {v:.0%}" for k, v in g.chosen.value_counts(normalize=True).items())
        rows.append(row)
    return pd.DataFrame(rows)


def plot(summ, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    truths = list(dict.fromkeys(summ.truth))
    models = list(MODELS) + ["aic"]
    fig, axes = plt.subplots(len(CENSORING), len(truths), figsize=(2.3 * len(truths), 5.2), sharey="row")
    for r, cens in enumerate(CENSORING):
        for cidx, truth in enumerate(truths):
            ax = axes[r, cidx]
            g = summ[(summ.truth == truth) & (summ.cens == cens) & (summ.n == 1000)].set_index("model")
            vals = [g.loc[m, "relerr_S0.01"] if m in g.index else np.nan for m in models]
            ax.bar(range(len(models)), vals, color=[f"C{i}" for i in range(len(models))])
            ax.set_yscale("log")
            ax.set_xticks(range(len(models)), ["Wei", "LN", "Burr", "ER", "GB2", "AIC"], fontsize=7, rotation=90)
            if r == 0:
                ax.set_title(truth, fontsize=9)
        axes[r, 0].set_ylabel(f"{cens}\nrel. RMSE of S at true 1%")
    fig.suptitle("Tail prediction (S = 0.01), n = 1000; lower is better", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.reps = 3
    OUT.mkdir(exist_ok=True)
    tag = "_quick" if args.quick else ""

    cens = {(f, l, s): censoring(f, true, s) for (f, l, true), s in itertools.product(TRUTHS, CENSORING)}
    t_s = {(f, l): [true_time(f, true, s) for s in S_LEVELS] for f, l, true in TRUTHS}
    tasks = [(f, l, true, n, s, cens[(f, l, s)], t_s[(f, l)], r)
             for (f, l, true), n, s, r in itertools.product(TRUTHS, SIZES, CENSORING, range(args.reps))]
    tasks.sort(key=lambda tk: tk[3], reverse=True)

    t0, rows = time.time(), []
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_dataset, tasks, chunksize=1), 1):
            rows.extend(res)
            if i % 400 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} datasets, {time.time() - t0:.0f}s", flush=True)
    df = add_aic_choice(pd.DataFrame(rows))
    df.to_csv(OUT / f"misspec_raw{tag}.csv", index=False)
    summ = summarise(df)
    summ.to_csv(OUT / f"misspec_summary{tag}.csv", index=False)
    plot(summ, OUT / f"misspec_tail{tag}.png")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
