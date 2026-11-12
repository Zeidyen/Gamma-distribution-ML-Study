"""Heads inside a neural network, on simulated patients with nonlinear covariate effects.

Five standard-normal covariates. The log scale (location) of every truth is

    f(x) = 0.5 sin(2 x1) + 0.4 x2 - 0.3 x3^2 + 0.2 x4 x5

Truths
  weibull_aft  Weibull, log scale f(x), shape 1.5
  lognormal    log T ~ N(f(x), 0.8^2)
  gb2_mixtail  GB2 with log a = f(x), c = 2, p = 1.5, and q = 0.75 when x1 > 0
               (heavy tail: S(t) ~ t^-1.5) but q = 6 otherwise (light tail).
               The tail itself depends on the patient.

Models: Weibull, log-normal, Burr XII, ER-BurrXII and GB2 (log-moment) heads, each with
shape parameters shared ("global") or predicted per patient ("x"); and
3-component Weibull, log-normal and Burr XII mixtures (all parameters from x). Same MLP
(64-64, ReLU, dropout 0.1) for all. 30% independent exponential censoring.

Because the truth is known, every patient's predicted survival curve is
compared with their true curve on 5000 uncensored test patients:
  excess log-loss      mean[log f_true - log f_hat]
  calibration at s     mean of S_hat at each patient's true S = s time, divided
                       by s (1 = calibrated; > 1 = too optimistic about survival)
  MAE at s             mean |S_hat - s| at those times, in probability units
  C-index              concordance of predicted median with the true time

Usage: .venv/bin/python sim_network.py [--reps 10] [--workers 15] [--quick]
"""

import argparse
import itertools
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from scipy import optimize, stats

from heads import DTYPE, HEADS, Head
from network import SurvivalNet, survival_quantile, train
from sim_recovery import OUT, _seed, default_start

TRUTHS = ("weibull_aft", "lognormal", "gb2_mixtail")
MODELS = [(f, s) for f in ("weibull", "lognormal", "burr12", "er_burr12", "maptb12", "gb2_lmom")
          for s in ("global", "x")] + [
    (m, "x") for m in ("mix_weibull", "mix_lognormal", "mix_burr12")]
SIZES = (1000, 5000)
N_FEATURES, N_TEST, CENS_TARGET = 5, 5000, 0.3
S_LEVELS = (0.5, 0.1, 0.01)


def f_loc(x):
    return 0.5 * np.sin(2 * x[:, 0]) + 0.4 * x[:, 1] - 0.3 * x[:, 2] ** 2 + 0.2 * x[:, 3] * x[:, 4]


def truth_raw(truth, x):
    """(family, raw [n, P]) of the true distribution for each row of x."""
    loc = f_loc(x)
    n = len(x)
    if truth == "weibull_aft":
        raw = np.column_stack([loc, np.full(n, np.log(1.5))])
        return "weibull", raw
    if truth == "lognormal":
        return "lognormal", np.column_stack([loc, np.full(n, np.log(0.8))])
    if truth == "gb2_mixtail":
        q = np.where(x[:, 0] > 0, 0.75, 6.0)
        return "gb2", np.column_stack([loc, np.full(n, np.log(2.0)),
                                       np.full(n, Head._shape_raw(1.5)), Head._shape_raw(q)])
    raise ValueError(truth)


def sample_truth(truth, x, rng):
    family, raw = truth_raw(truth, x)
    head = HEADS[family]()
    prm = {k: v.numpy() for k, v in head.params(torch.tensor(raw, dtype=DTYPE)).items()}
    return head.sample(rng, len(x), **prm)


def censoring_rate(truth):
    rng = np.random.default_rng(0)
    t = sample_truth(truth, rng.normal(size=(100_000, N_FEATURES)), rng)
    return optimize.brentq(lambda r: np.mean(-np.expm1(-r * t)) - CENS_TARGET, 1e-8, 1e6)


def make_test(truth):
    rng = np.random.default_rng([_seed(truth, "test"), 1])
    x = rng.normal(size=(N_TEST, N_FEATURES))
    t = sample_truth(truth, x, rng)
    family, raw = truth_raw(truth, x)
    head, raw = HEADS[family](), torch.tensor(raw, dtype=DTYPE)
    tt = torch.tensor(t, dtype=DTYPE)
    t_s = {s: survival_quantile(head, raw, s) for s in S_LEVELS}
    return dict(x=torch.tensor(x, dtype=DTYPE), t=tt, logf_true=head.log_pdf(tt, raw), t_s=t_s)


_TEST = {}


def run(task):
    truth, n, rep, family, shapes, rate = task
    torch.set_num_threads(1)
    torch.manual_seed(rep)
    if truth not in _TEST:
        _TEST[truth] = make_test(truth)
    test = _TEST[truth]

    rng = np.random.default_rng([_seed(truth, "train"), n, rep])
    x = rng.normal(size=(n, N_FEATURES))
    t = sample_truth(truth, x, rng)
    c = rng.exponential(1 / rate, size=n)
    e = (t <= c).astype(float)
    t = np.minimum(t, c)

    model = SurvivalNet(N_FEATURES, family, shapes=shapes)
    t0 = time.time()
    info = train(model, torch.tensor(x, dtype=DTYPE), torch.tensor(t, dtype=DTYPE),
                 torch.tensor(e, dtype=DTYPE), seed=rep,
                 raw0=model.head.init_raw(t) if hasattr(model.head, "init_raw") else default_start(model.head, t))
    row = dict(truth=truth, n=n, rep=rep, model=family, shapes=shapes, cens_obs=1 - e.mean(),
               seconds=time.time() - t0, **info)
    with torch.no_grad():
        raw_hat = model(test["x"])
        head = model.head
        row["excess_logloss"] = float((test["logf_true"] - head.log_pdf(test["t"], raw_hat)).mean())
        heavy = test["x"][:, 0] > 0
        for s in S_LEVELS:
            s_hat = torch.exp(head.log_surv(test["t_s"][s], raw_hat))
            row[f"cal_S{s}"] = float(s_hat.mean() / s)
            row[f"mae_S{s}"] = float((s_hat - s).abs().mean())
            if truth == "gb2_mixtail":  # heavy-tailed (x1 > 0) and light-tailed halves separately
                for name, m in (("heavy", heavy), ("light", ~heavy)):
                    row[f"cal_S{s}_{name}"] = float(s_hat[m].mean() / s)
        med = survival_quantile(head, raw_hat, 0.5)
    row["c_index"] = (stats.kendalltau(med.numpy(), test["t"].numpy()).statistic + 1) / 2
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.reps = 1
    tag = "_quick" if args.quick else ""
    OUT.mkdir(exist_ok=True)

    rates = {tr: censoring_rate(tr) for tr in TRUTHS}
    tasks = [(tr, n, r, f, s, rates[tr]) for tr, n, r, (f, s) in
             itertools.product(TRUTHS, SIZES, range(args.reps), MODELS)]
    tasks.sort(key=lambda tk: (tk[1], tk[3] == "gb2_lmom"), reverse=True)
    t0, rows = time.time(), []
    with Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(run, tasks), 1):
            rows.append(row)
            if i % 40 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} networks, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows).sort_values(["truth", "n", "model", "shapes", "rep"])
    df.to_csv(OUT / f"network_raw{tag}.csv", index=False)
    metrics = [c for c in df.columns if c.startswith(("excess", "cal_", "mae_", "c_index"))] + ["epochs", "seconds"]
    summ = df.groupby(["truth", "n", "model", "shapes"], sort=False)[metrics].mean().reset_index()
    summ.to_csv(OUT / f"network_summary{tag}.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
