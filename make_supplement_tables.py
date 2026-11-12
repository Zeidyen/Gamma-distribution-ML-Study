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
    t.to_csv(RES / "real_tests_all.csv", index=False)
    rows = []
    for m in REAL_ORDER[1:]:
        cells = []
        for ds, _ in DATASETS:
            r = t[(t.dataset == ds) & (t.comparator == m)].iloc[0]
            c = f"{r['diff']:.1f} [{r.lo:.1f}, {r.hi:.1f}]".replace("-", "$-$")
            cells.append(f"\\textbf{{{c}}}" if r.p < 0.05 else c)
        rows.append([real_name(m)] + cells)
    write("tab_tests", "Burr XII mixture minus every other model, integrated Brier score $\\times 1000$: mean "
          "[95\\% CI] from the corrected resampled $t$-test. Negative favours the Burr XII mixture; bold marks "
          "$p < 0.05$.", "stab:tests", "lrrrr", "Comparator & " + " & ".join(n for _, n in DATASETS), rows,
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
            cells.append(f"\\textbf{{{c}}}" if r.p < 0.05 else c)
        rows.append([real_name(m)] + cells)
    write("tab_kkbox_tests", "KKBox: Burr XII mixture minus every other model, mean [95\\% CI] over the 3 "
          "customer splits from the corrected resampled $t$-test (2 df). Bold marks $p < 0.05$.",
          "stab:kkbox_tests", "lrrr", "Comparator & IBS $\\times 1000$ & Test NLL & $C^{td}$", rows,
          size="\\scriptsize", note="Negative IBS and NLL differences, and positive $C^{td}$ differences, "
          "favour the Burr XII mixture.")
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    OUT = ap.parse_args().out
    OUT.mkdir(parents=True, exist_ok=True)
    recovery(); survival(); covariate(); misspec(); network(); real()
    t = tests()
    kt = kkbox()
    print(f"wrote {len(list(OUT.glob('*.tex')))} tables to {OUT}")
    print(t[t.p < 0.05][["dataset", "comparator", "diff", "p"]].to_string(index=False))
    print(kt.to_string(index=False))
