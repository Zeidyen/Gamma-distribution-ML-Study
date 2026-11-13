"""KKBox benchmark: the realdata.py models on a large dataset with heavy-tailed churn times.

Data: pycox `kkbox` (Kvamme & Borgan 2019): 2.81 million subscription spells from
2.27 million customers of the KKBox music-streaming service; event = churn
(no new subscription within 30 days of expiry); administrative censoring at
2017-01-29. Downloading needs Kaggle credentials and acceptance of the
competition rules (kkbox-churn-prediction-challenge).

Differences from realdata.py, all forced by size or structure:
  - splits are by customer (msno), so no customer is in both train and test;
    R repeats of a random customer split into train / validation / test
  - training rows can be subsampled (--n-train) for timing pilots
  - metrics that need full survival curves (concordance, Brier scores,
    D-calibration) use a fixed random subsample of --n-eval test rows; the test
    log-likelihood uses every test row
  - the random survival forest cannot hold ~2 million rows in memory (it stores
    a survival curve per leaf), so it is fitted on a --rsf-n-train subsample
  - fit time, prediction time (survival curves for the n_eval spells) and the
    peak memory added during fitting are recorded per fit
  - categorical covariates are one-hot coded for levels with >= 1% of spells,
    remaining levels pooled; booleans are 0/1
Models, network, optimiser and early stopping are those of realdata.py.

Usage: .venv/bin/python kkbox_benchmark.py --n-train 200000 --repeats 1 --models mix_burr12/x deepsurv
"""

import argparse
import datetime
import threading
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import psutil
import torch

import realdata as rd
from heads import DTYPE
from sim_recovery import OUT

pd.datetime = datetime.datetime  # pycox 0.3.0's kkbox code uses pd.datetime, removed in pandas 2

CATEGORICAL = ("payment_method_id", "city", "gender", "registered_via")
DROP = ("msno", "censor_duration")
_DATA = {}


def load_kkbox():
    from pycox.datasets import kkbox
    df = kkbox.read_df()
    groups = df.msno.astype("category").cat.codes.to_numpy()
    cols = []
    for c in df.columns:
        if c in DROP or c in ("duration", "event"):
            continue
        if c in CATEGORICAL:
            x = df[c].astype(str).fillna("nan")
            freq = x.value_counts(normalize=True)
            keep = freq.index[freq >= 0.01][1:]  # most common level is the reference
            cols += [(x == v).astype(float).rename(f"{c}={v}") for v in keep]
        else:
            cols.append(df[c].astype(float))
    X = pd.concat(cols, axis=1)
    out = X.assign(duration=df.duration.astype(float).to_numpy(), event=df.event.astype(float).to_numpy())
    return out, groups


def customer_splits(groups, repeats, test_frac=0.2, val_frac=0.1):
    """(repeat, train_idx, val_idx, test_idx), split by customer."""
    ids = np.unique(groups)
    for r in range(repeats):
        rng = np.random.default_rng(500 + r)
        u = rng.random(ids.max() + 1)
        tr = np.flatnonzero(u[groups] >= test_frac + val_frac)
        va = np.flatnonzero((u[groups] >= test_frac) & (u[groups] < test_frac + val_frac))
        te = np.flatnonzero(u[groups] < test_frac)
        yield r, tr, va, te


class PeakMemory:
    """Peak resident memory above the level at entry, sampled every 20 ms (MB)."""

    def __enter__(self):
        self.proc, self.peak, self.stop = psutil.Process(), 0, threading.Event()
        self.base = self.proc.memory_info().rss
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        while not self.stop.wait(0.02):
            self.peak = max(self.peak, self.proc.memory_info().rss)

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self.mb = max(self.peak - self.base, 0) / 2**20


def run(task):
    r, tr, va, te, ev_idx, model_name = task
    torch.set_num_threads(1)
    if "df" not in _DATA:
        _DATA["df"] = load_kkbox()[0]
    df = _DATA["df"]
    pre = rd.Preprocessor().fit(df.iloc[tr])
    X, Xv, Xte, Xev = (pre.transform(df.iloc[i]) for i in (tr, va, te, ev_idx))
    t, tv, t_te, t_ev = (df.duration.to_numpy()[i] for i in (tr, va, te, ev_idx))
    e, ev, e_te, e_ev = (df.event.to_numpy()[i] for i in (tr, va, te, ev_idx))
    seed = 7000 + r
    t0 = time.time()
    row = dict(dataset="kkbox", repeat=r, fold=0, model=model_name, n_train=len(tr), n_test=len(te),
               n_eval=len(ev_idx))
    try:
        with PeakMemory() as mem:
            if model_name in ("cox", "rsf"):
                model, info = rd.fit_classical(model_name, X, t, e, Xv, tv, ev, seed)
            elif model_name in ("deepsurv", "deephit"):
                model, info = rd.fit_pycox(model_name, X, t, e, Xv, tv, ev, seed)
            else:
                family, shapes = model_name.split("/")
                if family == "gb2_warm":
                    model, info = rd.fit_gb2_warm(X, t, e, Xv, tv, ev, seed)
                else:
                    model, info = rd.fit_parametric(family, shapes, X, t, e, Xv, tv, ev, seed)
        fit_s = time.time() - t0
        t1 = time.time()  # prediction: survival curves for the n_eval spells
        if model_name in ("cox", "rsf"):
            surv = rd.classical_surv_df(model_name, model, Xev)
        elif model_name in ("deepsurv", "deephit"):
            surv = rd.pycox_surv_df(model_name, model, Xev)
        else:
            surv = rd.parametric_surv_df(model, Xev, rd.surv_grid(t, t_ev))
        pred_s = time.time() - t1
        nll = np.nan
        if model_name not in ("cox", "rsf", "deepsurv", "deephit"):
            with torch.no_grad():
                nll = float(model.nll(torch.tensor(Xte, dtype=DTYPE), torch.tensor(t_te, dtype=DTYPE),
                                      torch.tensor(e_te, dtype=DTYPE)))
        row.update(seconds=time.time() - t0, fit_seconds=fit_s, predict_seconds=pred_s, fit_peak_mb=mem.mb,
                   epochs=info.get("epochs"), test_nll=nll, val_nll=info.get("best_val_nll"),
                   cox_penalty=info.get("penalty"), **rd.evaluate(surv, t_ev, e_ev), error=None)
    except Exception as exc:  # noqa: BLE001 - recorded in the results
        row.update(seconds=time.time() - t0, error=f"{type(exc).__name__}: {exc}")
    print(f"{model_name} repeat {r}: {row['seconds']:.0f}s, epochs {row.get('epochs')}, "
          f"ibs {row.get('ibs', float('nan')):.4f}, error {row['error']}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=None, help="subsample training rows (default: all)")
    ap.add_argument("--n-eval", type=int, default=50_000)
    ap.add_argument("--rsf-n-train", type=int, default=50_000, help="training subsample for the RSF")
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--models", nargs="*", default=rd.MODELS)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    df, groups = load_kkbox()
    print(f"KKBox: {len(df):,} spells, {len(np.unique(groups)):,} customers, {df.shape[1] - 2} features",
          flush=True)
    tasks = []
    for r, tr, va, te in customer_splits(groups, args.repeats):
        rng = np.random.default_rng(900 + r)
        if args.n_train and args.n_train < len(tr):
            tr = np.sort(rng.choice(tr, args.n_train, replace=False))
            va = np.sort(rng.choice(va, min(len(va), args.n_train // 4), replace=False))
        ev_idx = np.sort(rng.choice(te, min(args.n_eval, len(te)), replace=False))
        for m in args.models:
            if m == "rsf" and args.rsf_n_train < len(tr):
                rs = np.random.default_rng(1900 + r)
                tasks.append((r, np.sort(rs.choice(tr, args.rsf_n_train, replace=False)),
                              np.sort(rs.choice(va, min(len(va), args.rsf_n_train // 4), replace=False)),
                              te, ev_idx, m))
            else:
                tasks.append((r, tr, va, te, ev_idx, m))
    tasks.sort(key=lambda tk: ("gb2" in tk[5] or "deephit" in tk[5] or "mix" in tk[5]), reverse=True)
    # each finished fit is appended to a .part.jsonl (one JSON object per line), so a crashed run resumes where it stopped
    part = OUT / f"kkbox_raw{args.tag}.part.jsonl"
    done = set()
    if part.exists():
        done = set(map(tuple, pd.read_json(part, lines=True)[["repeat", "model"]].itertuples(index=False)))
        print(f"resuming: {len(done)} fits already in {part.name}", flush=True)
    tasks = [tk for tk in tasks if (tk[0], tk[5]) not in done]
    t0 = time.time()
    if tasks:
        with Pool(min(args.workers, len(tasks))) as pool:
            for row in pool.imap_unordered(run, tasks):
                with open(part, "a") as fh:
                    fh.write(pd.Series(row).to_json(double_precision=15) + "\n")
    out = pd.read_json(part, lines=True).sort_values(["model", "repeat"])
    path = OUT / f"kkbox_raw{args.tag}.csv"
    out.to_csv(path, index=False)
    print(f"done in {time.time() - t0:.0f}s -> {path}; errors: {out.error.notna().sum()}")


if __name__ == "__main__":
    main()
