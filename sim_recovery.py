"""Parameter-recovery and identifiability simulation for the likelihood heads.

Part A  intercept-only: simulate from each family at known parameters, fit by
        maximum likelihood, and record bias, RMSE, Wald coverage, failures and
        the conditioning of the observed information.
Part B  covariates: a linear map x -> raw (the simplest "network") with a
        known covariate effect on the scale only; checks the slopes are
        recovered and that no spurious effects appear on the shapes.

Usage:  .venv/bin/python sim_recovery.py [--reps 200] [--workers 15] [--quick]
Writes results/*.csv and results/*.png.
"""

import argparse
import itertools
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import optimize

from heads import DTYPE, HEADS, GB2GenF, GB2LogMoment

OUT = Path(__file__).resolve().parent / "results"

GB2_TRUE = {"A": dict(a=1.0, c=2.0, p=1.5, q=1.5), "B": dict(a=1.0, c=4.0, p=0.5, q=0.75)}

SETTINGS = [
    ("weibull", "W", dict(scale=1.0, shape=1.5)),
    ("burr12", "A", dict(a=1.0, c=3.0, k=1.0)),
    ("burr12", "B", dict(a=2.0, c=1.5, k=4.0)),
    ("er_burr12", "A", dict(a=1.0, c=3.0, k=1.5)),
    ("er_burr12", "B", dict(a=5.0, c=10.0, k=0.5)),  # region of Anafo et al. (2021) Table 2
    ("gb2", "A", GB2_TRUE["A"]),
    ("gb2", "B", GB2_TRUE["B"]),
    # the same GB2 distributions, fitted in two other parameterisations
    ("gb2_genf", "A", GB2GenF.from_gb2(**GB2_TRUE["A"])),
    ("gb2_genf", "B", GB2GenF.from_gb2(**GB2_TRUE["B"])),
    ("gb2_lmom", "A", GB2LogMoment.from_gb2(**GB2_TRUE["A"])),
    ("gb2_lmom", "B", GB2LogMoment.from_gb2(**GB2_TRUE["B"])),
]
REPARAM_OF_GB2 = ("gb2_genf", "gb2_lmom")
SURV_LEVELS = (0.5, 0.1, 0.01)  # fitted S(t) is checked at the true times where S = these
SIZES = (250, 1000, 4000)
CENSORING = (0.0, 0.3)
N_STARTS = 3
GRAD_TOL = 1e-5


def _seed(family, label):
    """Stable per-setting seed (str hash() changes between processes)."""
    return zlib.crc32(f"{family}/{label}".encode())


def data_spec(family, label, true):
    """Family and parameters that generate the data.

    Reparameterised GB2 heads are fitted to the very datasets the plain GB2
    head sees (same family name in the seed, same parameter values), so the
    comparison between parameterisations is like for like.
    """
    if family in REPARAM_OF_GB2:
        return "gb2", GB2_TRUE[label]
    return family, true


def surv_times(head, true):
    """True times t_s with S(t_s) = s for each s in SURV_LEVELS."""
    raw = torch.tensor(head.raw_from(**true), dtype=DTYPE)

    def gap(log_t, s):
        return float(head.log_surv(torch.tensor([np.exp(log_t)], dtype=DTYPE), raw)[0]) - np.log(s)

    return [float(np.exp(optimize.brentq(gap, -50, 50, args=(s,)))) for s in SURV_LEVELS]


def censoring_rate(head, true, target, seed=0):
    """Rate of exponential censoring that censors a fraction ``target``."""
    if target == 0:
        return 0.0
    t = head.sample(np.random.default_rng(seed), 200_000, **true)
    return optimize.brentq(lambda r: np.mean(-np.expm1(-r * t)) - target, 1e-8, 1e6)


def simulate(head, true, n, rate, rng):
    t = head.sample(rng, n, **true)
    if rate == 0:
        return t, np.ones(n)
    cens = rng.exponential(1 / rate, size=n)
    return np.minimum(t, cens), (t <= cens).astype(float)


def fit(nll, theta0, max_iter=500):
    """LBFGS from theta0; returns (theta, loss, grad norm)."""
    theta = torch.tensor(theta0, dtype=DTYPE, requires_grad=True)
    opt = torch.optim.LBFGS([theta], lr=1, max_iter=max_iter, tolerance_grad=1e-9,
                            tolerance_change=1e-12, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = nll(theta)
        if torch.isfinite(loss):
            loss.backward()
        return loss

    try:
        opt.step(closure)
    except (RuntimeError, IndexError):  # IndexError: torch strong-Wolfe line search hitting NaN losses
        return theta.detach().numpy(), np.inf, np.inf
    loss = nll(theta)
    if not torch.isfinite(loss):
        return theta.detach().numpy(), np.inf, np.inf
    (g,) = torch.autograd.grad(loss, theta)
    return theta.detach().numpy(), loss.item(), g.norm().item()


def fit_multistart(nll, theta0, rng):
    best = None
    for s in range(N_STARTS):
        start = theta0 if s == 0 else theta0 + rng.normal(0, 0.5, size=len(theta0))
        res = fit(nll, start)
        if best is None or res[1] < best[1]:
            best = res
    return best


def default_start(head, t):
    """Generic start: scale at the median observed time, shapes at 1."""
    theta = np.zeros(head.n_params)
    theta[0] = np.log(np.median(t))
    return theta


# ----------------------------------------------------------------- Part A

def run_intercept(task):
    family, label, true, n, cens, rate, t_s, rep = task
    torch.set_num_threads(1)
    head = HEADS[family]()
    data_family, data_true = data_spec(family, label, true)
    rng = np.random.default_rng([_seed(data_family, label), n, int(cens * 100), rep])
    t, e = simulate(HEADS[data_family](), data_true, n, rate, rng)
    tt, ee = torch.tensor(t, dtype=DTYPE), torch.tensor(e, dtype=DTYPE)

    def nll(theta):
        return head.nll(theta.expand(n, -1), tt, ee)

    t0 = time.time()
    theta, loss, gnorm = fit_multistart(nll, default_start(head, t), rng)
    row = dict(family=family, setting=label, n=n, cens_target=cens, cens_obs=1 - e.mean(), rep=rep,
               loss=loss, grad_norm=gnorm, seconds=time.time() - t0)
    truth = head.raw_from(**true)
    se = np.full(head.n_params, np.nan)
    cond = np.nan
    if np.isfinite(loss):
        H = torch.autograd.functional.hessian(nll, torch.tensor(theta, dtype=DTYPE)).numpy()
        evals = np.linalg.eigvalsh(H) if np.isfinite(H).all() else np.array([-1.0])
        if evals.min() > 0:
            cond = evals.max() / evals.min()
            se = np.sqrt(np.diag(np.linalg.inv(H * n)))
    row.update(cond=cond, converged=bool(np.isfinite(loss) and gnorm < GRAD_TOL and np.isfinite(cond)))
    for j in range(head.n_params):
        row[f"raw{j}"], row[f"raw{j}_true"], row[f"raw{j}_se"] = theta[j], truth[j], se[j]
    # natural-scale estimates of the named parameters
    est = head.params(torch.tensor(theta, dtype=DTYPE))
    for name in head.names:
        row[f"est_{name}"], row[f"true_{name}"] = float(est[name]), true[name]
    # parameterisation-free check: fitted S at the true times where S = 0.5, 0.1, 0.01
    s_hat = torch.exp(head.log_surv(torch.tensor(t_s, dtype=DTYPE), torch.tensor(theta, dtype=DTYPE)))
    for s, v in zip(SURV_LEVELS, s_hat.numpy()):
        row[f"S_hat_{s}"] = v if np.isfinite(loss) else np.nan
    return row


def run_intercept_safe(task):
    """run_intercept, but an unexpected error becomes a failed fit instead of ending the run."""
    try:
        return run_intercept(task)
    except Exception as exc:  # noqa: BLE001 - recorded in the row and counted as a failure
        family, label, true, n, cens, rate, t_s, rep = task
        return dict(family=family, setting=label, n=n, cens_target=cens, rep=rep, loss=np.inf,
                    grad_norm=np.inf, cond=np.nan, converged=False, error=f"{type(exc).__name__}: {exc}")


def summarise_intercept(df):
    rows = []
    for (fam, lab, n, cens), g in df.groupby(["family", "setting", "n", "cens_target"], sort=False):
        head = HEADS[fam]()
        ok = g[g.converged]
        base = dict(family=fam, setting=lab, n=n, cens=cens, reps=len(g),
                    fail_pct=100 * (1 - g.converged.mean()),
                    cond_median=ok.cond.median() if len(ok) else np.nan)
        # identifiability readout: largest |corr| between raw estimates across replicates
        raws = ok[[f"raw{j}" for j in range(head.n_params)]].to_numpy()
        base["max_abs_corr"] = (np.abs(np.corrcoef(raws.T) - np.eye(head.n_params)).max()
                                if len(ok) > 2 else np.nan)
        # one row per raw parameter: what the optimiser actually estimates.
        # rmse_raw is on the raw scale (log for positive parameters, as is for mu and Q).
        for j, (name, is_log) in enumerate(zip(head.raw_names, head.log_raw)):
            r, rt, rse = ok[f"raw{j}"], ok[f"raw{j}_true"], ok[f"raw{j}_se"]
            nat = np.exp if is_log else (lambda v: v)
            tru = nat(rt.iloc[0]) if len(ok) else np.nan
            rows.append(dict(base, param=name, true=tru, median_est=nat(r.median()),
                             rel_bias_median=(np.exp(r.median() - rt.iloc[0]) - 1) if is_log and len(ok) else np.nan,
                             bias_raw_median=r.median() - rt.iloc[0] if len(ok) else np.nan,
                             rmse_raw=np.sqrt(np.mean((r - rt) ** 2)),
                             coverage=np.mean(np.abs(r - rt) <= 1.96 * rse)))
        # natural parameters that are not themselves raw (ER-BurrXII k = q + 1/c); no Wald coverage
        for name in (nm for nm in head.names if nm not in head.raw_names):
            est, tru = ok[f"est_{name}"], ok[f"true_{name}"].iloc[0] if len(ok) else np.nan
            rows.append(dict(base, param=name, true=tru, median_est=est.median(),
                             rel_bias_median=est.median() / tru - 1,
                             bias_raw_median=np.log(est.median() / tru),
                             rmse_raw=np.sqrt(np.mean((np.log(est) - np.log(tru)) ** 2)),
                             coverage=np.nan))
    return pd.DataFrame(rows)


def summarise_survival(df):
    """Error of the fitted S(t) at the true S = 0.5, 0.1, 0.01 times.

    Invariant to parameterisation. ``all`` uses every fit with a finite
    likelihood, including those flagged as failed; ``converged`` only the rest.
    """
    rows = []
    for (fam, lab, n, cens), g in df.groupby(["family", "setting", "n", "cens_target"], sort=False):
        for subset, gg in (("all", g[np.isfinite(g.loss)]), ("converged", g[g.converged])):
            row = dict(family=fam, setting=lab, n=n, cens=cens, fits=subset, k=len(gg),
                       fail_pct=100 * (1 - g.converged.mean()))
            for s in SURV_LEVELS:
                err = gg[f"S_hat_{s}"] - s
                row[f"rmse_S{s}"] = np.sqrt(np.mean(err ** 2))
                row[f"max_abs_S{s}"] = err.abs().max()
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- Part B

COV_SETTINGS = [st for st in SETTINGS if st[1] in ("W", "A")]
COV_N, COV_CENS, COV_BETA = 2000, 0.3, np.array([0.5, -0.3])


def run_covariate(task):
    family, label, true, rate, rep = task
    torch.set_num_threads(1)
    head = HEADS[family]()
    data_family, data_true = data_spec(family, label, true)
    data_head = HEADS[data_family]()
    rng = np.random.default_rng([_seed(data_family, label), 99, rep])
    x = rng.normal(size=(COV_N, 2))
    # true scale varies with x; shapes are constant. For every head the first raw
    # output (log scale, log a or mu) shifts one-for-one with log scale.
    pars = dict(data_true)
    pars[data_head.names[0]] = data_true[data_head.names[0]] * np.exp(x @ COV_BETA)
    t = data_head.sample(rng, COV_N, **pars)
    if rate > 0:
        c = rng.exponential(1 / rate, size=COV_N)
        e = (t <= c).astype(float)
        t = np.minimum(t, c)
    else:
        e = np.ones(COV_N)
    X = torch.tensor(np.column_stack([np.ones(COV_N), x]), dtype=DTYPE)
    tt, ee = torch.tensor(t, dtype=DTYPE), torch.tensor(e, dtype=DTYPE)
    P = head.n_params

    def nll(theta):  # theta = vec of a [3, P] linear map (intercept + 2 slopes per raw output)
        return head.nll(X @ theta.view(3, P), tt, ee)

    start = np.zeros((3, P))
    start[0] = default_start(head, t)
    theta, loss, gnorm = fit_multistart(nll, start.ravel(), rng)
    B = theta.reshape(3, P)
    truth = np.zeros((3, P))
    truth[0] = head.raw_from(**true)
    truth[1:, 0] = COV_BETA
    return dict(family=family, setting=label, rep=rep, converged=np.isfinite(loss) and gnorm < GRAD_TOL,
                **{f"b{r}_{head.raw_names[j]}": B[r, j] for r in range(3) for j in range(P)},
                **{f"true_b{r}_{head.raw_names[j]}": truth[r, j] for r in range(3) for j in range(P)})


def summarise_covariate(df):
    rows = []
    for (fam, lab), g in df.groupby(["family", "setting"], sort=False):
        g = g.dropna(axis=1, how="all")
        ok = g[g.converged]
        for col in [c for c in g.columns if c.startswith("b") and c[1].isdigit()]:
            r = int(col[1])
            kind = "intercept (log param)" if r == 0 else f"slope x{r}"
            tru = ok[f"true_{col}"].iloc[0]
            rows.append(dict(family=fam, setting=lab, coef=col, kind=kind, true=tru,
                             mean_est=ok[col].mean(), bias=ok[col].mean() - tru,
                             rmse=np.sqrt(np.mean((ok[col] - tru) ** 2)),
                             fail_pct=100 * (1 - g.converged.mean())))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- plots

def plot_rmse(summ, path):
    """RMSE of each log estimate against n, one panel per setting; dashed = 30% censoring."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = list(summ.groupby(["family", "setting"], sort=False))
    ncol = 4
    nrow = int(np.ceil(len(groups) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.4 * nrow), sharey=True)
    axes = axes.ravel()
    ns = np.array(SIZES, dtype=float)
    for ax, ((fam, lab), g) in zip(axes, groups):
        for i, (param, gp) in enumerate(g.groupby("param", sort=False)):
            for cens, gg in gp.groupby("cens"):
                ax.plot(gg.n, gg.rmse_raw, marker="o", ms=3, color=f"C{i}",
                        ls="-" if cens == 0 else "--", label=param if cens == 0 else None)
        ref = g[g.n == ns[0]].rmse_raw.median()
        ax.plot(ns, ref * np.sqrt(ns[0] / ns), color="0.6", lw=1, ls=":", label="n^-1/2")
        ax.set(xscale="log", yscale="log", title=f"{fam} {lab}", xticks=ns,
               xticklabels=[f"{int(v)}" for v in ns], xlabel="n")
        ax.minorticks_off()
        ax.legend(fontsize=7, frameon=False, loc="lower left")
    for ax in axes[len(groups):]:
        ax.axis("off")
    for ax in axes[::ncol]:
        ax.set_ylabel("RMSE (raw scale)")
    fig.suptitle("Parameter recovery (solid: no censoring, dashed: 30% censoring)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--cov-reps", type=int, default=100)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--quick", action="store_true", help="tiny run to check the pipeline")
    args = ap.parse_args()
    if args.quick:
        args.reps, args.cov_reps = 4, 3
    OUT.mkdir(exist_ok=True)
    tag = "_quick" if args.quick else ""

    rates = {(f, l, c): censoring_rate(HEADS[data_spec(f, l, true)[0]](), data_spec(f, l, true)[1], c)
             for (f, l, true), c in itertools.product(SETTINGS, CENSORING)}
    t_s = {(f, l): surv_times(HEADS[f](), true) for f, l, true in SETTINGS}
    tasks = [(f, l, true, n, c, rates[(f, l, c)], t_s[(f, l)], r)
             for (f, l, true), n, c, r in itertools.product(SETTINGS, SIZES, CENSORING, range(args.reps))]
    # slowest (GB2-type, large n) first so the pool stays busy
    tasks.sort(key=lambda tk: (tk[0] in ("gb2", "er_burr12")) * tk[3], reverse=True)

    t0 = time.time()
    rows = []
    with Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(run_intercept_safe, tasks, chunksize=2), 1):
            rows.append(row)
            if i % 500 == 0 or i == len(tasks):
                print(f"part A {i}/{len(tasks)} fits, {time.time() - t0:.0f}s", flush=True)
    raw = pd.DataFrame(rows).sort_values(["family", "setting", "n", "cens_target", "rep"])
    raw.to_csv(OUT / f"recovery_raw{tag}.csv", index=False)
    order = {(f, l): i for i, (f, l, _) in enumerate(SETTINGS)}
    raw["_order"] = [order[(f, l)] for f, l in zip(raw.family, raw.setting)]
    summ = summarise_intercept(raw.sort_values(["_order", "n", "cens_target", "rep"]))
    summ.to_csv(OUT / f"recovery_summary{tag}.csv", index=False)
    summarise_survival(raw.sort_values(["_order", "n", "cens_target", "rep"])).to_csv(
        OUT / f"survival_summary{tag}.csv", index=False)
    plot_rmse(summ, OUT / f"recovery_rmse{tag}.png")

    cov_tasks = [(f, l, true, rates[(f, l, COV_CENS)], r)
                 for (f, l, true), r in itertools.product(COV_SETTINGS, range(args.cov_reps))]
    with Pool(args.workers) as pool:
        cov = pd.DataFrame(pool.map(run_covariate, cov_tasks, chunksize=1))
    cov.to_csv(OUT / f"covariate_raw{tag}.csv", index=False)
    summarise_covariate(cov).to_csv(OUT / f"covariate_summary{tag}.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
