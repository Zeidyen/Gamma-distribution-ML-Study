"""LaTeX tables for the supplementary material, built from results/*.csv.

Writes one .tex file per table to the manuscript folder (supp_tables/), and the
full set of corrected resampled t-tests to results/real_tests_all.csv.

The t-test is that of Nadeau and Bengio (2003): for the J = 25 paired fold
differences d_j, var = (1/J + n_test/n_train) s_d^2, with t on J - 1 df.
n_train is the training fold without the early-stopping split, as in Table 7
of the paper (this reproduces every value there).

Usage: .venv/bin/python make_supplement_tables.py [--out DIR]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RES = Path(__file__).parent / "results"
OUT = RES / "supp_tables"  # --out overrides; the manuscript reads latex/supp_tables/

FAMILY = {"weibull": "Weibull", "lognormal": "Log-normal", "burr12": "Burr XII",
          "er_burr12": "ER-BurrXII", "maptb12": "MAPT-Burr XII", "gb2": "GB2 $(a,c,p,q)$",
          "gb2_genf": "GB2, generalized F", "gb2_lmom": "GB2", "aic": "AIC",
          "mix_weibull": "Mixture, Weibull", "mix_lognormal": "Mixture, log-normal",
          "mix_burr12": "Mixture, Burr XII"}
REAL = {"gb2_lmom/x": "GB2 (x), cold start", "gb2_lmom/global": "GB2, cold start",
        "gb2_warm/x": "GB2 (x), warm start", "cox": "Cox (linear)",
        "rsf": "Random survival forest", "deepsurv": "DeepSurv", "deephit": "DeepHit"}
REAL_ORDER = ["mix_burr12/x", "mix_lognormal/x", "mix_weibull/x", "gb2_lmom/x", "gb2_lmom/global",
              "gb2_warm/x", "burr12/x", "burr12/global", "maptb12/x", "maptb12/global",
              "er_burr12/x", "er_burr12/global", "weibull/x", "weibull/global",
              "lognormal/x", "lognormal/global", "rsf", "cox", "deepsurv", "deephit"]
DATASETS = [("metabric", "METABRIC"), ("gbsg", "GBSG"), ("support", "SUPPORT"), ("flchain", "FLCHAIN")]
TRUTH = {"weibull W": "Weibull", "lognormal LN": "Log-normal", "burr12 A": "Burr XII A",
         "burr12 heavy": "Burr XII, heavy", "er_burr12 A": "ER-BurrXII A", "er_burr12 B": "ER-BurrXII B",
         "gb2 A": "GB2 A", "gb2 B": "GB2 B"}
NET_TRUTH = [("weibull_aft", "Weibull AFT"), ("lognormal", "Log-normal"), ("gb2_mixtail", "GB2, varying tail")]


def real_name(m):
    if m in REAL:
        return REAL[m]
    fam, shapes = m.split("/")
    return FAMILY[fam] + (" (x)" if shapes == "x" else "")


def f(x, d=3):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


def write(name, caption, label, colspec, header, rows, note=None, long=False, size="\\small"):
    """A booktabs table (or longtable) as a .tex file."""
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    if long:
        tex = (f"{{{size}\n\\begin{{longtable}}{{{colspec}}}\n\\caption{{{caption}}}\\label{{{label}}}\\\\\n"
               f"\\toprule\n{header} \\\\\n\\midrule\n\\endfirsthead\n"
               f"\\multicolumn{{{colspec.count('l') + colspec.count('r') + colspec.count('c')}}}{{l}}"
               f"{{\\emph{{Table \\ref{{{label}}}, continued}}}}\\\\\n\\toprule\n{header} \\\\\n\\midrule\n\\endhead\n"
               f"\\bottomrule\n\\endfoot\n{body}\n\\end{{longtable}}}}\n")
        if note:
            tex += f"\\vspace{{-0.5em}}\\noindent{{\\footnotesize {note}}}\n"
    else:
        tex = (f"\\begin{{table}}[htbp]\n\\centering\n\\caption{{{caption}}}\\label{{{label}}}\n{size}\n"
               f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n"
               f"\\bottomrule\n\\end{{tabular}}\n")
        if note:
            tex += f"\n\\par\\vspace{{0.3em}}{{\\footnotesize {note}}}\n"
        tex += "\\end{table}\n"
    (OUT / f"{name}.tex").write_text(tex)


def recovery():
    s = pd.read_csv(RES / "recovery_summary.csv")
    rows, last = [], None
    for (fam, st, n, cens), g in s.groupby(["family", "setting", "n", "cens"], sort=False):
        lab = f"{FAMILY[fam]} {st}" if (fam, st) != last else ""
        last = (fam, st)
        cov = g.coverage.dropna()
        rows.append([lab, f"{n:,}", f"{int(cens * 100)}", f(g.fail_pct.iloc[0], 1), f"{g.cond_median.iloc[0]:,.0f}"
                     if g.cond_median.iloc[0] >= 10 else f(g.cond_median.iloc[0], 1),
                     f(g.max_abs_corr.iloc[0], 2), f(100 * g.rel_bias_median.abs().max(), 1),
                     f"{cov.min():.2f}--{cov.max():.2f}" if len(cov) else "--"])
    write("tab_recovery", "Study 1: parameter recovery for every head, setting, sample size and censoring level "
          "(200 replicates per cell). Bias and coverage are over converged fits.", "stab:recovery",
          "llrrrrrrr", "Head and setting & $n$ & Cens.\\ \\% & Failed \\% & Cond.\\ no. & Max $|r|$ & "
          "Max $|$bias$|$ \\% & Coverage", rows,
          note="Cond.\\ no.: median condition number of the observed information. Max $|r|$: largest absolute "
          "correlation between log-parameter estimates across replicates. Max $|$bias$|$: largest median relative "
          "bias over the parameters. Coverage: range over parameters of 95\\% Wald interval coverage.", long=True)


def survival():
    s = pd.read_csv(RES / "survival_summary.csv")
    s = s[s.fits == "all"]
    rows, last = [], None
    for _, r in s.iterrows():
        lab = f"{FAMILY[r.family]} {r.setting}" if (r.family, r.setting) != last else ""
        last = (r.family, r.setting)
        rows.append([lab, f"{r.n:,}", f"{int(r.cens * 100)}", f(r.fail_pct, 1),
                     f(r["rmse_S0.5"], 4), f(r["rmse_S0.1"], 4), f(r["rmse_S0.01"], 4),
                     f(r["max_abs_S0.5"], 3)])
    write("tab_survival", "Study 1: error of the fitted survival function at the true times where "
          "$S = 0.5$, 0.1 and 0.01, over all fits including those flagged as failed.", "stab:survival",
          "llrrrrrr", "Head and setting & $n$ & Cens.\\ \\% & Fail \\% & RMSE $S_{0.5}$ & RMSE $S_{0.1}$ & "
          "RMSE $S_{0.01}$ & Max err.\\ $S_{0.5}$", rows, long=True, size="\\footnotesize")


def covariate():
    s = pd.read_csv(RES / "covariate_summary.csv")
    rows, last = [], None
    for _, r in s.iterrows():
        lab = f"{FAMILY[r.family]} {r.setting}" if (r.family, r.setting) != last else ""
        last = (r.family, r.setting)
        coef = r.coef.replace("_", r"\_")
        rows.append([lab, f"\\texttt{{{coef}}}", r.kind, f(r.true, 3), f(r.bias, 3), f(r.rmse, 3)])
    write("tab_covariate", "Study 1, linear-covariate check ($n = 2{,}000$, 30\\% censoring, 100 replicates). "
          "Every raw output is linear in $x_1, x_2$; only the log scale truly depends on them, so the true shape "
          "slopes are zero.", "stab:covariate", "lllrrr",
          "Head and setting & Coefficient & Kind & True & Bias & RMSE", rows, long=True)


def misspec():
    s = pd.read_csv(RES / "misspec_summary.csv")
    models = ["weibull", "lognormal", "burr12", "er_burr12", "gb2_lmom", "aic"]
    hdr = "True distribution & $n$ & " + " & ".join(FAMILY[m] for m in models)
    for metric, mname, dig in [("excess_logloss", "excess test log-loss", 3), ("relerr_S0.01",
                               "relative error of $S$ at the true 1\\% time", 2)]:
        for cens, cname in [("random30", "30\\% random censoring"), ("admin15", "follow-up ending when 15\\% survive")]:
            rows, last = [], None
            for truth in TRUTH:
                for n in (250, 1000, 4000):
                    g = s[(s.truth == truth) & (s.n == n) & (s.cens == cens)].set_index("model")
                    vals = [g.loc[m, metric] for m in models]
                    best = np.nanmin(vals[:-1])
                    cells = [("\\textbf{" + f(v, dig) + "}") if np.isclose(v, best) and m != "aic" else f(v, dig)
                             for m, v in zip(models, vals)]
                    rows.append([TRUTH[truth] if truth != last else "", f"{n:,}"] + cells)
                    last = truth
            tag = f"{'ll' if metric == 'excess_logloss' else 'tail'}_{cens}"
            write(f"tab_misspec_{tag}", f"Study 2: {mname}, {cname} (mean over 100 datasets). Bold: best single head.",
                  f"stab:misspec_{tag}", "lr" + "r" * len(models), hdr, rows)
    rows = []
    for truth in TRUTH:
        for cens in ("random30", "admin15"):
            g = s[(s.truth == truth) & (s.n == 1000) & (s.cens == cens) & (s.model == "aic")]
            ch = g.chosen.iloc[0]
            for k, v in FAMILY.items():
                ch = ch.replace(k + " ", v + " ") if k in ("gb2_lmom", "er_burr12") else ch
            for k in ("burr12", "weibull", "lognormal"):
                ch = ch.replace(k + " ", FAMILY[k] + " ")
            ch = ch.replace("%", "\\%")
            rows.append([TRUTH[truth], "random 30\\%" if cens == "random30" else "ends at 15\\%", ch])
    write("tab_aic", "Study 2: heads selected by AIC, $n = 1{,}000$ (percentage of 100 datasets).", "stab:aic",
          "llp{9cm}", "True distribution & Censoring & Selected head", rows)


def network():
    s = pd.read_csv(RES / "network_summary.csv")
    order = s[["model", "shapes"]].drop_duplicates()
    for metric, mname, dig, tag in [("excess_logloss", "excess test log-loss", 3, "ll"),
                                    ("cal_S0.01", "tail calibration (mean predicted $S$ at each patient's "
                                     "true 1\\% time, divided by 0.01; 1 is calibrated)", 2, "cal"),
                                    ("c_index", "concordance of predicted median with true time", 3, "c")]:
        rows = []
        for _, (m, sh) in order.iterrows():
            cells = []
            for n in (1000, 5000):
                for t, _ in NET_TRUTH:
                    v = s[(s.model == m) & (s.shapes == sh) & (s.truth == t) & (s.n == n)][metric]
                    cells.append(f(v.iloc[0], dig) if len(v) else "--")
            rows.append([FAMILY[m] + (" (x)" if sh == "x" else "")] + cells)
        hdr = ("& \\multicolumn{3}{c}{$n = 1{,}000$} & \\multicolumn{3}{c}{$n = 5{,}000$} \\\\\n"
               "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\nHead & " + " & ".join(x for _, x in NET_TRUTH) * 1
               + " & " + " & ".join(x for _, x in NET_TRUTH))
        write(f"tab_network_{tag}", f"Study 3: {mname}, median of 20 runs.", f"stab:network_{tag}",
              "lrrrrrr", hdr, rows, note="(x): shape parameters predicted per patient.", size="\\scriptsize")
    rows = []
    g = s[s.truth == "gb2_mixtail"]
    for _, (m, sh) in order.iterrows():
        cells = []
        for n in (1000, 5000):
            r = g[(g.model == m) & (g.shapes == sh) & (g.n == n)]
            cells += [f(r["cal_S0.01_heavy"].iloc[0], 2), f(r["cal_S0.01_light"].iloc[0], 2)]
        rows.append([FAMILY[m] + (" (x)" if sh == "x" else "")] + cells)
    hdr = ("& \\multicolumn{2}{c}{$n = 1{,}000$} & \\multicolumn{2}{c}{$n = 5{,}000$} \\\\\n"
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nHead & Heavy tail & Light tail & Heavy tail & Light tail")
    write("tab_network_halves", "Study 3, GB2 truth whose tail varies: tail calibration at the 1\\% level in the "
          "heavy-tailed ($x_1 > 0$, $q = 0.75$) and light-tailed ($q = 6$) halves of the test patients.",
          "stab:network_halves", "lrrrr", hdr, rows, note="(x): shape parameters predicted per patient.")


def real():
    s = pd.read_csv(RES / "real_summary.csv")
    raw = pd.read_csv(RES / "real_raw.csv")
    secs = raw.groupby(["dataset", "model"]).seconds.mean()
    for ds, dname in DATASETS:
        g = s[s.dataset == ds].set_index("model")
        rows = []
        for m in REAL_ORDER:
            r = g.loc[m]
            nll = f(r.test_nll_mean, 3) if not m.startswith(("cox", "rsf", "deep")) else "--"
            rows.append([real_name(m), f(r.c_td_mean, 3), f(1000 * r.ibs_mean, 1), f(1000 * r.ibll_mean, 1),
                         f(1000 * r.brier_q50_mean, 1), f(1000 * r.brier_q75_mean, 1), f(1000 * r.brier_q90_mean, 1),
                         f"{r.dcal_pass_pct:.0f}", nll, f(secs[(ds, m)], 1)])
        write(f"tab_real_{ds}", f"{dname}: all metrics, mean over 25 test folds.", f"stab:real_{ds}",
              "lrrrrrrrrr", "Model & $C^{td}$ & IBS & IBLL & $B_{50}$ & $B_{75}$ & $B_{90}$ & D-cal \\% & "
              "Test NLL & Seconds", rows, size="\\footnotesize",
              note="IBS, IBLL and Brier scores $\\times 1000$ (lower is better). $B_{50}$, $B_{75}$, $B_{90}$: "
              "Brier score at the 50th, 75th and 90th percentiles of test times. D-cal \\%: folds passing "
              "D-calibration at 5\\%. Test NLL: held-out censored negative log-likelihood per patient "
              "(parametric models only). Seconds: mean fitting time per fold. (x): shapes predicted per patient.")


def holm(p):
    """Holm step-down adjusted p-values."""
    p = np.asarray(p, dtype=float)
    adj, run = np.empty(len(p)), 0.0
    for i, j in enumerate(np.argsort(p)):
        run = max(run, (len(p) - i) * p[j])
        adj[j] = min(run, 1.0)
    return adj


def mark(c, r):
    """Bold for p < 0.05; a dagger when it survives Holm adjustment as well."""
    if r.p >= 0.05:
        return c
    return f"\\textbf{{{c}}}" + ("$^{\\dagger}$" if r.p_holm < 0.05 else "")


def tests():
    raw = pd.read_csv(RES / "real_raw.csv")
    out = []
    for ds, _ in DATASETS:
        a = raw[(raw.dataset == ds) & (raw.model == "mix_burr12/x")].set_index(["repeat", "fold"])
        ratio = (a.n_test / a.n_train).mean()
        for m in REAL_ORDER[1:]:
            b = raw[(raw.dataset == ds) & (raw.model == m)].set_index(["repeat", "fold"])
            d = ((a.ibs - b.ibs) * 1000).dropna()
            J = len(d)
            se = np.sqrt((1 / J + ratio) * d.var(ddof=1))
            tq = stats.t.ppf(0.975, J - 1)
            out.append(dict(dataset=ds, comparator=m, folds=J, diff=d.mean(), lo=d.mean() - tq * se,
                            hi=d.mean() + tq * se, p=2 * stats.t.sf(abs(d.mean() / se), J - 1)))
    t = pd.DataFrame(out)
    t["p_holm"] = t.groupby("dataset").p.transform(holm)  # over the 19 comparators per dataset
    t.to_csv(RES / "real_tests_all.csv", index=False)
    rows = []
    for m in REAL_ORDER[1:]:
        cells = []
        for ds, _ in DATASETS:
            r = t[(t.dataset == ds) & (t.comparator == m)].iloc[0]
            c = f"{r['diff']:.1f} [{r.lo:.1f}, {r.hi:.1f}]".replace("-", "$-$")
            cells.append(mark(c, r))
        rows.append([real_name(m)] + cells)
    write("tab_tests", "Burr XII mixture minus every other model, integrated Brier score $\\times 1000$: mean "
          "[95\\% CI] from the corrected resampled $t$-test. Negative favours the Burr XII mixture; bold marks "
          "$p < 0.05$, and $^{\\dagger}$ marks $p < 0.05$ after Holm adjustment over the 19 comparators.", "stab:tests", "lrrrr", "Comparator & " + " & ".join(n for _, n in DATASETS), rows,
          size="\\scriptsize")
    return t


def kkbox_raw():
    """KKBox results from the three runs: main run, RSF on its subsample, and the Weibull mixture
    rerun after the non-finite-gradient fix in network.train (the main run's Weibull mixture
    weights went NaN in the first epoch). Also written to results/kkbox_raw.csv."""
    full = pd.read_csv(RES / "kkbox_raw_full.csv")
    raw = pd.concat([full[full.model != "mix_weibull/x"], pd.read_csv(RES / "kkbox_raw_rsf.csv"),
                     pd.read_csv(RES / "kkbox_raw_mixweibull_fixed.csv")], ignore_index=True)
    raw.to_csv(RES / "kkbox_raw.csv", index=False)
    return raw


def kkbox_tests(raw, metrics=(("ibs", 1000), ("test_nll", 1), ("c_td", 1))):
    """Corrected resampled t-test, Burr XII mixture minus each model, over the 3 customer splits."""
    a = raw[raw.model == "mix_burr12/x"].set_index("repeat")
    ratio = (a.n_test / a.n_train).mean()  # full-data ratio; RSF trained on a subsample but is scored the same way
    out = []
    for m in REAL_ORDER[1:]:
        b = raw[raw.model == m].set_index("repeat")
        for k, scale in metrics:
            d = ((a[k] - b[k]) * scale).dropna()
            if len(d) < 2:
                continue
            J = len(d)
            se = np.sqrt((1 / J + ratio) * d.var(ddof=1))
            tq = stats.t.ppf(0.975, J - 1)
            out.append(dict(comparator=m, metric=k, splits=J, diff=d.mean(), lo=d.mean() - tq * se,
                            hi=d.mean() + tq * se, p=2 * stats.t.sf(abs(d.mean() / se), J - 1),
                            burr_better=int((d < 0).sum() if k != "c_td" else (d > 0).sum())))
    t = pd.DataFrame(out)
    t["p_holm"] = t.groupby("metric").p.transform(holm)  # over the comparators for each metric
    t.to_csv(RES / "kkbox_tests.csv", index=False)
    return t


def kkbox():
    raw = kkbox_raw()
    s = raw.groupby("model").mean(numeric_only=True)
    rows = []
    for m in REAL_ORDER:
        r = s.loc[m]
        rows.append([real_name(m), f(r.c_td, 3), f(1000 * r.ibs, 1), f(1000 * r.ibll, 1), f(1000 * r.brier_q50, 1),
                     f(1000 * r.brier_q75, 1), f(1000 * r.brier_q90, 1), f"{r.dcal_stat:,.0f}",
                     f(r.test_nll, 3), f"{r.seconds:,.0f}"])
    write("tab_real_kkbox", "KKBox: all metrics, mean over 3 customer-level splits.", "stab:real_kkbox",
          "lrrrrrrrrr", "Model & $C^{td}$ & IBS & IBLL & $B_{50}$ & $B_{75}$ & $B_{90}$ & D-cal $\\chi^2$ & "
          "Test NLL & Seconds", rows, size="\\footnotesize",
          note="As Tables~\\ref{stab:real_metabric}--\\ref{stab:real_flchain}, except: concordance, Brier scores and D-calibration use a fixed random "
          "50,000 test spells per split (test NLL uses all of them); every model fails D-calibration at this "
          "size, so the $\\chi^2$ statistic (9 df) is shown instead of a pass rate; the random survival forest "
          "was trained on 50,000 spells. The Weibull mixture is from the rerun after the gradient fix "
          "(Section~\\ref{sec:nettrain}). Seconds: mean fitting time per split.")
    t = kkbox_tests(raw)
    rows = []
    for m in REAL_ORDER[1:]:
        cells = []
        for k in ("ibs", "test_nll", "c_td"):
            r = t[(t.comparator == m) & (t.metric == k)]
            if r.empty:
                cells.append("--")
                continue
            r = r.iloc[0]
            d = 1 if k == "ibs" else 3
            c = f"{r['diff']:.{d}f} [{r.lo:.{d}f}, {r.hi:.{d}f}]".replace("-", "$-$")
            cells.append(mark(c, r))
        rows.append([real_name(m)] + cells)
    write("tab_kkbox_tests", "KKBox: Burr XII mixture minus every other model, mean [95\\% CI] over the 3 "
          "customer splits from the corrected resampled $t$-test (2 df). Bold marks $p < 0.05$, and $^{\\dagger}$ marks "
          "$p < 0.05$ after Holm adjustment over the comparators for that metric.",
          "stab:kkbox_tests", "lrrr", "Comparator & IBS $\\times 1000$ & Test NLL & $C^{td}$", rows,
          size="\\scriptsize", note="Negative IBS and NLL differences, and positive $C^{td}$ differences, "
          "favour the Burr XII mixture.")
    return t


def sci(x):
    """1.2e-10 as $1.2 \\times 10^{-10}$; 0 as 0."""
    if x == 0:
        return "0"
    m, e = f"{x:.1e}".split("e")
    return f"${m} \\times 10^{{{int(e)}}}$"


def betainc():
    s = pd.read_csv(RES / "betainc_accuracy_summary.csv").set_index("band")
    rows = []
    for band, lab in (("I > 1e-10", "$I > 10^{-10}$"), ("1e-300 < I < 1e-10", "$10^{-300} < I \\le 10^{-10}$"),
                      ("I < 1e-300", "$I \\le 10^{-300}$"), ("all", "All")):
        r = s.loc[band]
        rows.append([lab, f"{int(r.points):,}", sci(r.median_err), sci(r.max_err), f"{int(r.points - r.scipy_finite)}",
                     sci(r.dlogx_max), sci(r.da_max), sci(r.db_max)])
    write("tab_betainc", "Accuracy of the differentiable $\\log I_x(a, b)$ against an 80-digit mpmath reference: "
          "1,300 points with $a, b \\in \\{0.05, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 990\\}$ and 13 values of $x$ from "
          "$10^{-8}$ to 0.999.", "stab:betainc", "lrrrrrrr",
          "$I$ (reference) & Points & \\multicolumn{2}{c}{Error of $\\log I$} & SciPy & "
          "\\multicolumn{3}{c}{Max.\\ relative error of derivative} \\\\\n & & Median & Max. & zero & "
          "$\\partial/\\partial \\log x$ & $\\partial/\\partial a$ & $\\partial/\\partial b$", rows, size="\\footnotesize",
          note="Error of $\\log I$: absolute error, equal to the relative error of $I$. SciPy zero: points where "
          "\\texttt{scipy.special.betainc} returns 0, so that $\\log I = -\\infty$. Derivatives: autograd against "
          "numerical differentiation of the reference, as relative error, or absolute error where the derivative "
          "is below $10^{-8}$. Because $1 - I_x(a, b) = I_{1-x}(b, a)$, the grid covers both tails. Source: "
          "\\texttt{betainc\\_accuracy.py}.")


def kkbox_tail():
    fits = pd.read_csv(RES / "kkbox_tail.csv")
    km = pd.read_csv(RES / "kkbox_km.csv")
    best = fits.aic.min()
    rows = []
    for _, r in fits.iterrows():
        rows.append([r.model, f(r.loglik_per_spell, 4).replace("-", "$-$"), f"{(r.aic - best):,.0f}", f(r.S_759, 3)])
    rows.append(["Kaplan--Meier", "--", "--", f(km.S.iloc[-1], 3)])
    write("tab_kkbox_tail", "KKBox: distributions fitted without covariates to all 2,814,735 spells.",
          "stab:kkbox_tail", "lrrr", "Model & Log-likelihood per spell & $\\Delta$AIC & $S(759)$", rows,
          note="$S(759)$: fitted survival at the end of follow-up (759 days). Burr XII: $a = "
          f"{fits.set_index('model').loc['Burr XII', 'a']:.1f}$, $c = {fits.set_index('model').loc['Burr XII', 'c']:.2f}$, "
          f"$k = {fits.set_index('model').loc['Burr XII', 'k']:.3f}$. GB2 (log-moment form) reached the shape cap "
          f"($p = {fits.set_index('model').loc['GB2', 'p']:.0f}$). Source: \\texttt{{kkbox\\_tail.py}}.")


def cap_sensitivity():
    """Study 2 rerun with the shape cap at 100 to 100,000 (sim_cap_sensitivity.py)."""
    d = pd.read_csv(RES / "cap_sensitivity_raw.csv")
    d = d[d.model.isin(["er_burr12", "gb2_lmom"])].copy()
    d["shape"] = np.where(d.model == "gb2_lmom", d[["par_p", "par_q"]].max(axis=1), d.par_k)
    key = ["model", "truth", "n", "cens", "rep"]
    ref = d[d.cap == 1e3].set_index(key)[["excess_logloss", "S_hat_0.01"]]
    d = d.join(ref, on=key, rsuffix="_ref")
    rows = []
    for (m, cap), g in d.groupby(["model", "cap"]):
        dS = (g["S_hat_0.01"] - g["S_hat_0.01_ref"]).abs()
        same = ((g.excess_logloss - g.excess_logloss_ref).abs() < 1e-3).mean()
        rows.append([FAMILY[m] if cap == 1e2 else "", f"{cap:,.0f}", f(100 * (1 - g.converged.mean()), 1),
                     f(100 * (g["shape"] >= cap / 2).mean(), 1), f(g.excess_logloss.median(), 4),
                     f(g.excess_logloss.mean(), 3), f(100 * same, 1), f(dS.max(), 4)])
    write("tab_cap", "Study 2 with the shape cap at 100, 1,000 (as in the paper), 10,000 and 100,000: "
          "$n = 250$ and 1,000, both censoring schemes, all eight true distributions, 100 datasets each "
          "(3,200 fits per row).", "stab:cap", "lrrrrrrr",
          "Head & Cap & Failed (\\%) & At cap (\\%) & \\multicolumn{2}{c}{Excess log-loss} & Unchanged (\\%) & "
          "Max.\\ $|\\Delta S|$ \\\\\n & & & & Median & Mean & &", rows, size="\\footnotesize",
          note="Failed: gradient norm above the convergence tolerance at the end of fitting (Section~\\ref{sec:s2}). "
          "At cap: largest capped shape ($p$ or $q$ for GB2, $k$ for ER-BurrXII) at least half the cap. "
          "Unchanged: fits whose excess log-loss is within 0.001 of the fit to the same dataset with the cap at 1,000. "
          "Max.\\ $|\\Delta S|$: largest absolute change, over all 3,200 datasets, of the predicted survival at the "
          "true 1\\% time, against the cap at 1,000. Every dataset uses the same random starts at every cap. "
          "Source: \\texttt{sim\\_cap\\_sensitivity.py}.")


def kkbox_n50k():
    """KKBox with every model trained on the same 50,000 spells, with fit and prediction costs."""
    raw = pd.read_csv(RES / "kkbox_raw_n50k.csv")
    s = raw.groupby("model").mean(numeric_only=True)
    rows = []
    for m in REAL_ORDER:
        r = s.loc[m]
        rows.append([real_name(m), f(r.c_td, 3), f(1000 * r.ibs, 1), f(r.test_nll, 2), f"{r.fit_seconds:,.0f}",
                     f(r.predict_seconds, 1), f"{r.fit_peak_mb:,.0f}", "--" if pd.isna(r.epochs) else f"{r.epochs:.0f}"])
    write("tab_kkbox_n50k", "KKBox with every model trained on the same random 50,000 spells per split "
          "(mean over the 3 customer splits of Table~\\ref{stab:real_kkbox}).", "stab:kkbox_n50k", "lrrrrrrr",
          "Model & $C^{td}$ & IBS & Test NLL & Fit (s) & Predict (s) & Memory (MB) & Epochs", rows,
          size="\\footnotesize",
          note="Validation set for early stopping: a random 12,500 of the validation spells. Concordance and IBS use "
          "a fixed random 50,000 test spells per split, and test NLL all of the about 562,000 test spells. "
          "Predict: time to compute survival curves for the 50,000 evaluation spells. Memory: peak resident memory "
          "during fitting above that at the start of the fit, sampled every 20 ms; memory already held by the "
          "worker from earlier fits is not counted, so small values are lower bounds. Three fits ran at a time on "
          "an Apple M4 Max (16 cores, 48 GB), one thread each. Weibull (x): test NLL 332.6, 4.0 and 3.6 on the "
          "three splits. Source: \\texttt{kkbox\\_benchmark.py --n-train 50000}.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    OUT = ap.parse_args().out
    OUT.mkdir(parents=True, exist_ok=True)
    recovery(); survival(); covariate(); misspec(); network(); real()
    t = tests()
    kt = kkbox()
    betainc(); kkbox_tail(); cap_sensitivity(); kkbox_n50k()
    print(f"wrote {len(list(OUT.glob('*.tex')))} tables to {OUT}")
    print(t[t.p < 0.05][["dataset", "comparator", "diff", "p"]].to_string(index=False))
    print(kt.to_string(index=False))
