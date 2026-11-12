"""Real-data benchmark: parametric heads vs DeepSurv, DeepHit and DSM-style mixtures.

Datasets (pycox): METABRIC, SUPPORT, GBSG, FLCHAIN. Each is split by
stratified 5-fold cross-validation, repeated 5 times (25 test folds). Inside
each training fold 20% is held out (stratified) for early stopping; every
model sees exactly the same train / validation / test rows.

Every model uses the same MLP (64-64, ReLU, dropout 0.1), Adam lr 1e-3,
batch 256, early stopping with patience 20 (max 400 epochs). No tuning.

Models
  ours       Weibull, log-normal, Burr XII, ER-BurrXII, GB2 (log-moment) heads; shape
             parameters shared ("global") or per patient ("x")
             gb2_warm: GB2 (a, c, p, q) started from the fitted Burr XII network
  DSM-style  3-component Weibull or log-normal mixtures, all parameters from
             the network (Nagpal et al. 2021), trained like ours; plus a
             3-component Burr XII mixture
  DeepSurv   pycox CoxPH (Katzman et al. 2018), Breslow baseline hazard
  DeepHit    pycox DeepHitSingle, 50 equidistant time bins, alpha 0.2, sigma 0.1
  Cox        linear Cox PH (lifelines), ridge penalty chosen from 0.001-1 by
             validation log-likelihood, then refitted; Breslow baseline
  RSF        random survival forest (scikit-survival; 200 trees, min leaf 15,
             sqrt features); Cox and RSF do not early-stop, so they are
             fitted on the training + validation rows

Metrics on each test fold (pycox EvalSurv, Kaplan-Meier censoring weights)
  c_td        Antolini time-dependent concordance
  ibs, ibll   integrated Brier score / binomial log-likelihood on a grid from
              the smallest test time to the 90th percentile of test times
              (beyond that the IPCW weights become unstable)
  brier_q50/75/90  Brier score at the 50/75/90th percentiles of test times
  dcal_stat, dcal_p  D-calibration (Haider et al. 2020), 10 bins, chi-square
  test_nll    mean censored negative log-likelihood (parametric models only)

Usage: .venv/bin/python realdata.py [--repeats 5] [--workers 15] [--quick] [--models ...]
"""

import argparse
import itertools
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.model_selection import StratifiedKFold, train_test_split

from heads import DTYPE, Head
from network import SurvivalNet, train
from sim_recovery import OUT, default_start

warnings.filterwarnings("ignore")

# pycox 0.3.0 calls scipy.integrate.simps(scores, time_grid); SciPy >= 1.14
# removed simps, and its replacement simpson takes x only as a keyword
import scipy.integrate as _si  # noqa: E402

if not hasattr(_si, "simps"):
    _si.simps = lambda y, x=None, **kw: _si.simpson(y, x=x, **kw)

DATASETS = ("metabric", "support", "gbsg", "flchain")
PARAMETRIC = [(f, s) for f in ("weibull", "lognormal", "burr12", "er_burr12", "maptb12", "gb2_lmom")
              for s in ("global", "x")] + [
    ("gb2_warm", "x"), ("mix_weibull", "x"), ("mix_lognormal", "x"), ("mix_burr12", "x")]
MODELS = [f"{f}/{s}" for f, s in PARAMETRIC] + ["deepsurv", "deephit", "cox", "rsf"]
HIDDEN, DROPOUT, LR, BATCH, MAX_EPOCHS, PATIENCE = (64, 64), 0.1, 1e-3, 256, 400, 20
N_FOLDS = 5


# ----------------------------------------------------------------- data

def load(name):
    """DataFrame with feature columns plus 'duration' and 'event'."""
    from pycox import datasets
    if name == "flchain":
        # pycox 0.3.0's loader expects an 'Unnamed: 0' column that Rdatasets now
        # calls 'rownames'; same processing as pycox otherwise
        if not datasets.flchain.path.exists():
            try:
                datasets.flchain.read_df()  # downloads the file, then fails in processing
            except KeyError:
                pass
        df = pd.read_feather(datasets.flchain.path)
        df = (df.drop(columns=["chapter", "rownames"])
              .loc[lambda d: d["creatinine"].notna()].reset_index(drop=True)
              .assign(sex=lambda d: (d["sex"] == "M").astype(float))
              .rename(columns={"futime": "duration", "death": "event"}))
        df = pd.get_dummies(df, columns=["sample.yr", "flc.grp"], drop_first=True, dtype=float)
    else:
        df = getattr(datasets, name).read_df()
    df = df.astype(float)
    # a few durations are exactly 0; log t needs t > 0
    pos = df.duration[df.duration > 0]
    df.loc[df.duration <= 0, "duration"] = pos.min() / 2
    return df


class Preprocessor:
    """Binary columns as 0/1, 3-10 level columns one-hot, others standardised (fit on train).

    Continuous columns with skewness > 2 and no negative values (lab values,
    node counts, receptor levels) are log1p-transformed first. Without this,
    a few test patients with extreme kappa / lambda in FLCHAIN gave the linear
    Cox model linear predictors near -1000 and reversed its ranking.
    """

    def fit(self, df):
        self.cols = [c for c in df.columns if c not in ("duration", "event")]
        self.kind, self.levels, self.mean, self.sd = {}, {}, {}, {}
        for c in self.cols:
            u = np.unique(df[c])
            if len(u) <= 2:
                self.kind[c] = "bin"
                self.levels[c] = u.max()
            elif len(u) <= 10:
                self.kind[c] = "cat"
                self.levels[c] = u[1:]
            else:
                x = df[c].to_numpy()
                self.kind[c] = "log" if (stats.skew(x) > 2 and x.min() >= 0) else "num"
                v = np.log1p(x) if self.kind[c] == "log" else x
                self.mean[c], self.sd[c] = v.mean(), v.std(ddof=1) or 1.0
        return self

    def transform(self, df):
        out = []
        for c in self.cols:
            if self.kind[c] == "bin":
                out.append((df[c].to_numpy() == self.levels[c]).astype(float)[:, None])
            elif self.kind[c] == "cat":
                out.append(np.column_stack([(df[c].to_numpy() == v).astype(float) for v in self.levels[c]]))
            else:
                v = df[c].to_numpy()
                v = np.log1p(np.clip(v, 0, None)) if self.kind[c] == "log" else v
                out.append(((v - self.mean[c]) / self.sd[c])[:, None])
        return np.hstack(out)


def splits(df, repeats):
    """(repeat, fold, train_idx, val_idx, test_idx); val is a stratified 20% of the training fold."""
    for r in range(repeats):
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=r)
        for k, (tr, te) in enumerate(skf.split(df, df.event)):
            tr, va = train_test_split(tr, test_size=0.2, stratify=df.event.iloc[tr], random_state=100 + r * 10 + k)
            yield r, k, tr, va, te


# ----------------------------------------------------------------- models

def surv_grid(t_train, t_test):
    grid = np.linspace(0, t_train.max(), 400)
    return np.unique(np.concatenate([grid, t_test]))


def fit_parametric(family, shapes, X, t, e, Xv, tv, ev, seed):
    torch.manual_seed(seed)  # network initialisation
    model = SurvivalNet(X.shape[1], family, hidden=HIDDEN, dropout=DROPOUT, shapes=shapes)
    head = model.head
    raw0 = head.init_raw(t) if hasattr(head, "init_raw") else default_start(head, t)
    T = lambda a: torch.tensor(a, dtype=DTYPE)  # noqa: E731
    info = train(model, T(X), T(t), T(e), lr=LR, batch=BATCH, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                 seed=seed, raw0=raw0, val=(T(Xv), T(tv), T(ev)))
    return model, info


def gb2_from_burr(burr, n_features):
    """A GB2 (a, c, p, q) network that reproduces a fitted Burr XII network (p = 1, q = k)."""
    gb = SurvivalNet(n_features, "gb2", hidden=HIDDEN, dropout=DROPOUT, shapes="x")
    gb.body.load_state_dict(burr.body.state_dict())
    with torch.no_grad():
        W, b = burr.out.weight, burr.out.bias
        gb.out.weight.zero_()
        for dst, src in ((0, 0), (1, 1), (3, 2)):  # log a, log c, q <- k
            gb.out.weight[dst] = W[src]
            gb.out.bias[dst] = b[src]
        gb.out.bias[2] = float(Head._shape_raw(1.0))  # p = 1
    gb.eval()
    return gb


def fit_gb2_warm(X, t, e, Xv, tv, ev, seed):
    """GB2 started from a fitted Burr XII network.

    Burr XII(a, c, k) is GB2(a, c, p=1, q=k). The GB2 network copies the
    Burr body and its a, c, k output rows (k -> q; the shape cap changes q by
    under 1% for k < 10) and starts with p = 1 for everyone, so it begins at
    the Burr solution. Early stopping keeps the starting point as a candidate,
    so on the validation set GB2 can only match or improve on Burr XII.
    """
    burr, info_b = fit_parametric("burr12", "x", X, t, e, Xv, tv, ev, seed)
    gb = gb2_from_burr(burr, X.shape[1])
    T = lambda a: torch.tensor(a, dtype=DTYPE)  # noqa: E731
    info = train(gb, T(X), T(t), T(e), lr=LR, batch=BATCH, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                 seed=seed, raw0=None, val=(T(Xv), T(tv), T(ev)))
    info["epochs"] += info_b["epochs"]
    info["burr_val_nll"] = info_b["best_val_nll"]
    return gb, info


@torch.no_grad()
def parametric_surv_df(model, Xte, grid, chunk=200):
    raw = model(torch.tensor(Xte, dtype=DTYPE))
    g = torch.tensor(grid, dtype=DTYPE)
    rows = []
    for k in range(0, len(g), chunk):
        gk = g[k:k + chunk]
        rows.append(torch.exp(model.head.log_surv(gk[:, None].expand(-1, len(raw)),
                                                  raw[None].expand(len(gk), -1, -1))).numpy())
    return pd.DataFrame(np.vstack(rows), index=grid)


def fit_pycox(kind, X, t, e, Xv, tv, ev, seed):
    import torchtuples as tt
    from pycox.models import CoxPH, DeepHitSingle
    torch.manual_seed(seed)
    f32 = lambda a: a.astype("float32")  # noqa: E731
    if kind == "deepsurv":
        net = tt.practical.MLPVanilla(X.shape[1], list(HIDDEN), 1, batch_norm=False, dropout=DROPOUT)
        model = CoxPH(net, tt.optim.Adam(lr=LR))
        y, yv = (f32(t), f32(e)), (f32(tv), f32(ev))
    else:
        lab = DeepHitSingle.label_transform(50)
        y = lab.fit_transform(t, e)
        yv = lab.transform(tv, ev)
        net = tt.practical.MLPVanilla(X.shape[1], list(HIDDEN), lab.out_features, batch_norm=False,
                                      dropout=DROPOUT)
        model = DeepHitSingle(net, tt.optim.Adam(lr=LR), alpha=0.2, sigma=0.1, duration_index=lab.cuts)
    log = model.fit(f32(X), y, BATCH, MAX_EPOCHS, [tt.callbacks.EarlyStopping(patience=PATIENCE)],
                    verbose=False, val_data=(f32(Xv), yv))
    if kind == "deepsurv":
        model.compute_baseline_hazards()
    return model, dict(epochs=len(log.monitors["train_"].scores["loss"]["score"]))


COX_PENALTIES = (0.001, 0.01, 0.1, 1.0)


def fit_classical(kind, X, t, e, Xv, tv, ev, seed):
    """Linear Cox PH (lifelines, ridge) or random survival forest (scikit-survival).

    Cox: the ridge penalty is chosen from COX_PENALTIES by validation partial
    log-likelihood (the counterpart of the networks' early stopping), then the
    model is refitted on training + validation rows. RSF: fitted on training
    + validation rows with fixed settings.
    """
    cols = [f"x{j}" for j in range(X.shape[1])]
    if kind == "cox":
        from lifelines import CoxPHFitter
        frame = lambda A, tt, ee: pd.DataFrame(A, columns=cols).assign(duration=tt, event=ee)  # noqa: E731
        scores = {}
        for pen in COX_PENALTIES:
            try:
                m = CoxPHFitter(penalizer=pen).fit(frame(X, t, e), "duration", "event")
                scores[pen] = m.score(frame(Xv, tv, ev), scoring_method="log_likelihood")
            except Exception:  # noqa: BLE001 - a penalty that does not converge is skipped
                continue
        best = max(scores, key=scores.get)
        model = CoxPHFitter(penalizer=best).fit(frame(np.vstack([X, Xv]), np.concatenate([t, tv]),
                                                      np.concatenate([e, ev])), "duration", "event")
        return model, {"penalty": best}
    else:
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.util import Surv
        model = RandomSurvivalForest(n_estimators=200, min_samples_leaf=15, max_features="sqrt",
                                     n_jobs=1, random_state=seed).fit(
            np.vstack([X, Xv]), Surv.from_arrays(np.concatenate([e, ev]).astype(bool), np.concatenate([t, tv])))
    return model, {}


def classical_surv_df(kind, model, Xte):
    if kind == "cox":
        return model.predict_survival_function(pd.DataFrame(Xte, columns=[f"x{j}" for j in range(Xte.shape[1])]))
    s = model.predict_survival_function(Xte, return_array=True)  # [n_test, n_times]
    return pd.DataFrame(s.T, index=model.unique_times_)


def pycox_surv_df(kind, model, Xte):
    Xte = Xte.astype("float32")
    if kind == "deephit":
        return model.interpolate(10).predict_surv_df(Xte)
    return model.predict_surv_df(Xte)


# ----------------------------------------------------------------- metrics

def surv_at_own_time(surv, t_te):
    """S_i(t_i) from a step-function survival DataFrame (index = times)."""
    idx = np.searchsorted(surv.index.to_numpy(), t_te, side="right") - 1
    vals = surv.to_numpy()
    s = np.where(idx >= 0, vals[np.clip(idx, 0, None), np.arange(len(t_te))], 1.0)
    return np.clip(s, 0, 1)


def d_calibration(s_own, e_te, bins=10):
    """Haider et al. (2020): S_i(t_i) should be uniform; censored rows spread over [0, S_i]."""
    edges = np.linspace(0, 1, bins + 1)
    counts = np.zeros(bins)
    for s, ev in zip(s_own, e_te):
        b = min(int(s * bins), bins - 1)
        if ev:
            counts[b] += 1
        elif s <= 0:
            counts[0] += 1
        else:
            counts[b] += (s - edges[b]) / s
            counts[:b] += (edges[1] - edges[0]) / s
    expected = len(s_own) / bins
    stat = ((counts - expected) ** 2 / expected).sum()
    return stat, 1 - stats.chi2.cdf(stat, bins - 1)


def evaluate(surv, t_te, e_te):
    from pycox.evaluation import EvalSurv
    ev = EvalSurv(surv, t_te, e_te.astype(int), censor_surv="km")
    q = np.quantile(t_te, [0.5, 0.75, 0.9])
    grid = np.linspace(t_te.min(), q[2], 100)
    out = dict(c_td=ev.concordance_td("antolini"), ibs=ev.integrated_brier_score(grid),
               ibll=ev.integrated_nbll(grid))
    for name, b in zip(("brier_q50", "brier_q75", "brier_q90"), ev.brier_score(q).to_numpy()):
        out[name] = b
    out["dcal_stat"], out["dcal_p"] = d_calibration(surv_at_own_time(surv, t_te), e_te)
    return out


# ----------------------------------------------------------------- driver

_DATA = {}


def run(task):
    name, r, k, tr, va, te, model_name = task
    torch.set_num_threads(1)
    if name not in _DATA:
        _DATA[name] = load(name)
    df = _DATA[name]
    pre = Preprocessor().fit(df.iloc[tr])
    X, Xv, Xte = (pre.transform(df.iloc[i]) for i in (tr, va, te))
    t, tv, t_te = (df.duration.to_numpy()[i] for i in (tr, va, te))
    e, ev, e_te = (df.event.to_numpy()[i] for i in (tr, va, te))
    seed = 1000 * r + k
    t0 = time.time()
    row = dict(dataset=name, repeat=r, fold=k, model=model_name, n_train=len(tr), n_test=len(te))
    try:
        if model_name in ("cox", "rsf"):
            model, info = fit_classical(model_name, X, t, e, Xv, tv, ev, seed)
            surv = classical_surv_df(model_name, model, Xte)
            nll = np.nan
        elif model_name in ("deepsurv", "deephit"):
            model, info = fit_pycox(model_name, X, t, e, Xv, tv, ev, seed)
            surv = pycox_surv_df(model_name, model, Xte)
            nll = np.nan
        else:
            family, shapes = model_name.split("/")
            if family == "gb2_warm":
                model, info = fit_gb2_warm(X, t, e, Xv, tv, ev, seed)
            else:
                model, info = fit_parametric(family, shapes, X, t, e, Xv, tv, ev, seed)
            surv = parametric_surv_df(model, Xte, surv_grid(t, t_te))
            with torch.no_grad():
                nll = float(model.nll(torch.tensor(Xte, dtype=DTYPE), torch.tensor(t_te, dtype=DTYPE),
                                      torch.tensor(e_te, dtype=DTYPE)))
        row.update(seconds=time.time() - t0, epochs=info.get("epochs"), test_nll=nll,
                   val_nll=info.get("best_val_nll"), burr_val_nll=info.get("burr_val_nll"),
                   cox_penalty=info.get("penalty"),
                   **evaluate(surv, t_te, e_te), error=None)
    except Exception as exc:  # noqa: BLE001 - recorded in the results
        row.update(seconds=time.time() - t0, error=f"{type(exc).__name__}: {exc}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--quick", action="store_true", help="one fold of METABRIC, all models")
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--models", nargs="*", default=None,
                    help="run only these models and merge them into the existing results/real_raw.csv")
    args = ap.parse_args()
    tag = "_quick" if args.quick else ""
    OUT.mkdir(exist_ok=True)

    tasks = []
    for name in (["metabric"] if args.quick else args.datasets):
        df = load(name)  # also downloads once, before the workers start
        for r, k, tr, va, te in splits(df, 1 if args.quick else args.repeats):
            if args.quick and k > 0:
                break
            tasks += [(name, r, k, tr, va, te, m) for m in (args.models or MODELS)]
    # big datasets and slow heads first
    tasks.sort(key=lambda tk: (len(tk[3]), "gb2" in tk[6] or "mix" in tk[6]), reverse=True)
    t0, rows = time.time(), []
    with Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(run, tasks), 1):
            rows.append(row)
            if i % 30 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} fits, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    if args.models and (OUT / f"real_raw{tag}.csv").exists():  # merge, replacing earlier rows of these models
        old = pd.read_csv(OUT / f"real_raw{tag}.csv")
        df = pd.concat([old[~old.model.isin(args.models)], df], ignore_index=True)
    df = df.sort_values(["dataset", "model", "repeat", "fold"])
    df.to_csv(OUT / f"real_raw{tag}.csv", index=False)
    metrics = ["c_td", "ibs", "ibll", "brier_q50", "brier_q75", "brier_q90", "dcal_stat", "test_nll"]
    summ = df.groupby(["dataset", "model"])[metrics].agg(["mean", "std"])
    summ.columns = [f"{m}_{s}" for m, s in summ.columns]
    summ["dcal_pass_pct"] = df.groupby(["dataset", "model"]).dcal_p.apply(lambda p: 100 * (p > 0.05).mean())
    summ.reset_index().to_csv(OUT / f"real_summary{tag}.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s; errors: {df.error.notna().sum()}")


if __name__ == "__main__":
    main()
