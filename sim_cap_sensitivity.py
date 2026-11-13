"""Sensitivity of Study 2 to the shape cap SHAPE_MAX (default 1,000).

Reruns the misspecification study (sim_misspec.py) at n = 250 and 1,000, both
censoring schemes and 100 datasets per cell, with the cap at 100, 1,000, 10,000
and 100,000. Only the GB2 (log-moment) and ER-BurrXII heads use the cap; the
other heads are refitted too, so that each dataset draws the same random starts
as in sim_misspec.py. The cap = 1,000 rows therefore reproduce
results/misspec_raw.csv. Each row also records the fitted parameters, to show
whether the cap binds.

Usage: .venv/bin/python sim_cap_sensitivity.py [--workers 14]
"""

import argparse
import itertools
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

import heads
import sim_misspec as sm
from heads import DTYPE, HEADS
from sim_recovery import OUT, _seed, default_start, fit_multistart, GRAD_TOL

CAPS = (1e2, 1e3, 1e4, 1e5)
SIZES = (250, 1000)
CAPPED = ("er_burr12", "gb2_lmom")


def run_dataset(task):
    """sim_misspec.run_dataset with the cap set and the fitted parameters kept."""
    cap, family, label, true, n, scheme, cens, t_s, rep = task
    if heads.SHAPE_MAX != cap:  # the true GB2 and ER-BurrXII also go through the cap: rebuild cached test sets
        heads.SHAPE_MAX = cap
        sm._TEST_CACHE.clear()
    torch.set_num_threads(1)
    rng = np.random.default_rng([_seed(family, label), n, sm.CENSORING.index(scheme), rep, 11])
    t = HEADS[family]().sample(rng, n, **true)
    kind, value = cens
    c = rng.exponential(1 / value, size=n) if kind == "rate" else np.full(n, value)
    e = (t <= c).astype(float)
    t = np.minimum(t, c)
    tt, ee = torch.tensor(t, dtype=DTYPE), torch.tensor(e, dtype=DTYPE)
    t_test, logf_true = sm.test_set(family, label, true)

    rows = []
    for model in sm.MODELS:
        head = HEADS[model]()
        par = {}
        t0 = time.time()
        try:
            def nll(theta):
                return head.nll(theta.expand(n, -1), tt, ee)

            theta, loss, gnorm = fit_multistart(nll, default_start(head, t), rng)
            th = torch.tensor(theta, dtype=DTYPE)
            s_hat = torch.exp(head.log_surv(torch.tensor(t_s, dtype=DTYPE), th)).numpy()
            with torch.no_grad():
                kl = float((logf_true - head.log_pdf(t_test, th.expand(N_TEST, -1))).mean())
                par = {f"par_{k}": float(v) for k, v in head.params(th).items()}
            err = None
        except Exception as exc:  # noqa: BLE001 - recorded, counted as a failed fit
            loss, gnorm, s_hat, kl, err = np.inf, np.inf, np.full(len(sm.S_LEVELS), np.nan), np.nan, repr(exc)
        if model not in CAPPED and cap != 1e3:
            continue  # uncapped heads: identical at every cap, keep one copy
        row = dict(cap=cap, truth=f"{family} {label}", n=n, cens=scheme, rep=rep, model=model, loss=loss,
                   grad_norm=gnorm, converged=bool(np.isfinite(loss) and gnorm < GRAD_TOL),
                   excess_logloss=kl, seconds=time.time() - t0, error=err, **par)
        for s, v in zip(sm.S_LEVELS, s_hat):
            row[f"S_hat_{s}"] = v
        rows.append(row)
    return rows


N_TEST = sm.N_TEST


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    cens = {(f, l, s): sm.censoring(f, true, s) for (f, l, true), s in itertools.product(sm.TRUTHS, sm.CENSORING)}
    t_s = {(f, l): [sm.true_time(f, true, s) for s in sm.S_LEVELS] for f, l, true in sm.TRUTHS}
    tasks = [(cap, f, l, true, n, s, cens[(f, l, s)], t_s[(f, l)], r)
             for cap, (f, l, true), n, s, r in itertools.product(CAPS, sm.TRUTHS, SIZES, sm.CENSORING,
                                                                  range(args.reps))]
    tasks.sort(key=lambda tk: tk[4], reverse=True)
    # each finished dataset is appended to a .part.jsonl (one JSON object per line), so a crashed run resumes where it stopped
    part = OUT / "cap_sensitivity_raw.part.jsonl"
    if part.exists():
        done = set(map(tuple, pd.read_json(part, lines=True)[["cap", "truth", "n", "cens", "rep"]]
                       .drop_duplicates().itertuples(index=False)))
        tasks = [tk for tk in tasks if (tk[0], f"{tk[1]} {tk[2]}", tk[4], tk[5], tk[8]) not in done]
        print(f"resuming: {len(done)} datasets already in {part.name}", flush=True)
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_dataset, tasks, chunksize=1), 1):
            with open(part, "a") as fh:
                fh.writelines(pd.Series(r).to_json(double_precision=15) + "\n" for r in res)
            if i % 800 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} datasets, {time.time() - t0:.0f}s", flush=True)
    df = pd.read_json(part, lines=True)
    df.to_csv(OUT / "cap_sensitivity_raw.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
