"""Accuracy of heads.log_betainc and its gradients against an 80-digit mpmath reference.

Grid: shapes a, b in {0.05, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 990} and 13 values
of x from 1e-8 to 0.999 (1,300 points). For each point it records log I_x(a, b)
from heads.log_betainc, from SciPy (log of scipy.special.betainc) and from
mpmath, and the derivatives of log I with respect to log x, a and b from
autograd and from mpmath (numerical differentiation at 80 digits). Because
1 - I_x(a, b) = I_{1-x}(b, a), the grid covers both tails of the survival
function. Writes results/betainc_accuracy.csv and prints a summary.

Usage: .venv/bin/python betainc_accuracy.py
"""

import itertools
from multiprocessing import Pool

import mpmath as mp
import numpy as np
import pandas as pd
import torch
from scipy import special

from heads import DTYPE, log_betainc
from sim_recovery import OUT

SHAPES = (0.05, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 990)
XS = (1e-8, 1e-6, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999)
mp.mp.dps = 80


def reference(point):
    a, b, x = (mp.mpf(v) for v in point)

    def logI(a_, b_, logx_):
        return mp.log(mp.betainc(a_, b_, 0, mp.exp(logx_), regularized=True))

    lx = mp.log(x)
    return (float(logI(a, b, lx)), float(mp.diff(lambda v: logI(a, b, v), lx)),
            float(mp.diff(lambda v: logI(v, b, lx), a)), float(mp.diff(lambda v: logI(a, v, lx), b)))


def main():
    pts = list(itertools.product(SHAPES, SHAPES, XS))
    with Pool(14) as pool:
        ref = np.array(pool.map(reference, pts, chunksize=8))
    a, b, x = (torch.tensor(np.array(v, dtype=float), dtype=DTYPE, requires_grad=True) for v in zip(*pts))
    # as in the heads, log x and log(1 - x) both come from the same tensor, so the gradient
    # with respect to log x flows through both arguments
    log_x = torch.log(x.detach()).requires_grad_(True)
    log_1mx = torch.log1p(-torch.exp(log_x))
    out = log_betainc(log_x, log_1mx, a, b)
    g = torch.autograd.grad(out.sum(), (log_x, a, b))
    an, bn, xn = (np.array(v, dtype=float) for v in zip(*pts))
    with np.errstate(divide="ignore"):
        scipy_log = np.log(special.betainc(an, bn, xn))
    df = pd.DataFrame(dict(a=an, b=bn, x=xn, log_I_ref=ref[:, 0], log_I=out.detach().numpy(), log_I_scipy=scipy_log,
                           dlogx_ref=ref[:, 1], dlogx=g[0].numpy(), da_ref=ref[:, 2], da=g[1].numpy(),
                           db_ref=ref[:, 3], db=g[2].numpy()))
    df["err"] = (df.log_I - df.log_I_ref).abs()  # absolute error in log I = relative error in I
    df["err_scipy"] = (df.log_I_scipy - df.log_I_ref).abs()
    for k in ("dlogx", "da", "db"):
        # relative error; absolute error where the derivative itself is below 1e-8 (log I ~ 0)
        df[f"err_{k}"] = (df[k] - df[f"{k}_ref"]).abs() / np.maximum(df[f"{k}_ref"].abs(), 1e-8)
    df.to_csv(OUT / "betainc_accuracy.csv", index=False)

    bands = [("I > 1e-10", df.log_I_ref > np.log(1e-10)),
             ("1e-300 < I < 1e-10", (df.log_I_ref <= np.log(1e-10)) & (df.log_I_ref > np.log(1e-300))),
             ("I < 1e-300", df.log_I_ref <= np.log(1e-300))]
    rows = []
    for name, m in bands + [("all", df.log_I_ref.notna())]:
        g_ = df[m]
        rows.append(dict(band=name, points=len(g_), median_err=g_.err.median(), max_err=g_.err.max(),
                         scipy_finite=int(np.isfinite(g_.log_I_scipy).sum()),
                         scipy_max_err=g_.err_scipy[np.isfinite(g_.log_I_scipy)].max(),
                         **{f"{k}_median": g_[f"err_{k}"].median() for k in ("dlogx", "da", "db")},
                         **{f"{k}_max": g_[f"err_{k}"].max() for k in ("dlogx", "da", "db")}))
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "betainc_accuracy_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.to_string(index=False))
    print(df.sort_values("err", ascending=False).head(5)[["a", "b", "x", "log_I_ref", "err"]].to_string(index=False))
    for k in ("dlogx", "da", "db"):
        print(df.sort_values(f"err_{k}", ascending=False).head(3)[["a", "b", "x", "log_I_ref", f"{k}_ref", k,
                                                                     f"err_{k}"]].to_string(index=False))


if __name__ == "__main__":
    main()
