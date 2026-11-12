# Heavy-tailed likelihood heads for deep survival analysis

Code for the paper *Heavy-tailed likelihood heads for deep survival analysis:
Burr XII, generalized beta and their mixtures* (A. Y. Anafo, submitted to the
Journal of Statistical Computation and Simulation).

## Reproducing the paper

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q tests            # 87 tests, ~20 s
.venv/bin/python sim_recovery.py               # Study 1 (Table 3, Fig. 1)
.venv/bin/python sim_misspec.py                # Study 2 (Table 4, Fig. 2)
.venv/bin/python sim_network.py                # Study 3 (Table 5, Fig. 3)
.venv/bin/python realdata.py                   # benchmark (Tables 6-7, Fig. 4); downloads data via pycox
.venv/bin/python make_paper_figures.py         # paper figures -> results/paper_figures/
.venv/bin/python make_supplement_tables.py     # supplement tables + all t-tests -> results/
```

KKBox (Section 3.5, Table 8, Tables S19-S20) needs a Kaggle API key in
`~/.kaggle/kaggle.json` and acceptance of the rules of the
`kkbox-churn-prediction-challenge` competition; pycox then downloads and builds
the data (about 2.8 million rows; the build needs several GB of memory). About
2 hours with 7 workers, plus a few minutes for the forest:

```bash
M=$(.venv/bin/python -c "import realdata as r; print(' '.join(m for m in r.MODELS if m != 'rsf'))")
.venv/bin/python kkbox_benchmark.py --repeats 3 --workers 7 --models $M --tag _full
.venv/bin/python kkbox_benchmark.py --repeats 3 --workers 3 --models rsf --tag _rsf
.venv/bin/python kkbox_benchmark.py --repeats 3 --workers 3 --models mix_weibull/x --tag _mixweibull_fixed
```

`make_supplement_tables.py` merges the three into `results/kkbox_raw.csv`. The
`_full` run predates the non-finite-gradient check in `network.train`, which is
why the Weibull mixture has its own run; with the check in place, the `_full`
command alone should give the same Weibull mixture (same seeds and splits).

All results the paper reports are saved in `results/` (`*_raw.csv`: one row
per fit or fold; `*_summary.csv`: aggregated). The notes below are the
development log kept while the study was run.

---

## Development notes: deep GB2-family survival heads

Likelihood heads for parametric deep survival models: a network outputs the
parameters of a heavy-tailed lifetime distribution for each subject, trained on
the right-censored log-likelihood. This folder is weeks 1–2 of the plan: the
heads themselves, plus a simulation that checks parameter recovery and
measures identifiability.

## Files

| File | What it is |
|---|---|
| `heads.py` | Weibull, Burr XII, ER-BurrXII and GB2 heads (log f, log S, sampler, censored NLL) and a differentiable `log_betainc` |
| `tests/test_heads.py` | Density integrates to 1, S equals the tail integral of f, sampler passes KS, `log_betainc` matches SciPy, gradcheck |
| `sim_recovery.py` | Part A intercept-only recovery and Part B linear-covariate recovery |
| `sim_misspec.py` | Misspecification: every head fitted to data from every true family |
| `plot_survival.py` | Figure of S(t) prediction error from `survival_summary.csv` |
| `results/` | `recovery_raw.csv` (one row per fit), `recovery_summary.csv`, `recovery_rmse.png`, `covariate_*.csv`, `run.log` |

## Setup and running

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python torch scipy numpy pandas matplotlib pytest
.venv/bin/python -m pytest -q tests          # ~10 s
.venv/bin/python sim_recovery.py --quick     # pipeline check, ~20 s
.venv/bin/python sim_recovery.py             # full run (200 reps), 15 workers
```

## Design decisions

- **Parameterisation.** Every parameter is `exp(raw)`, so the network output is
  the log parameter. ER-BurrXII uses raw = (log a, log c, log q) with
  q = k − 1/c, so the constraint k > 1/c (finite Burr mean) holds for every
  raw value; k is reported as q + 1/c.
- **ER-BurrXII is implemented as GB2(a, c, 1/c, k − 1/c).** The tests check it
  against eq. 2.3 of Anafo, Brew & Nasiru (2021).
- **GB2 survival** is the regularized incomplete beta I_u(q, p), u = 1/(1+(t/a)^c).
  PyTorch has no `betainc`, so `log_betainc` uses the Numerical Recipes
  continued fraction in torch (autograd differentiates through it), switching
  to the complement above x* = (a+1)/(a+b+2) so small survival probabilities
  are never computed by cancellation. It stops once all elements converge,
  which made it about 10× faster than a fixed 200 iterations.
- **Shape cap (added after the first misspecification run).** GB2-type shape
  parameters are smoothly capped below 1e3 (p = e^r / (1 + e^r/1e3)); the
  generalized F has P ≥ 2e-3. Log-normal, Weibull and generalized gamma are
  limits of GB2 with p and/or q → ∞, so on such data the MLE runs to infinity.
  Without the cap, fits reached p ≈ 1e13, the lgamma / continued-fraction
  arithmetic broke down, the fitted "density" integrated to 1e133, and the
  optimiser reported impossible likelihoods (excess test log-loss of −125). A
  test now checks normalisation at extreme raw values. All results below
  were rerun with the cap.
- **Float64 throughout.**
- **Fitting** is full-batch LBFGS with strong-Wolfe line search from a generic
  start (log scale at the median observed time, shapes at 1) plus two jittered
  restarts; the best of three is kept. A fit counts as converged if the
  gradient norm of the mean NLL is below 1e-5 and the observed information is
  positive definite.
- **Censoring** is independent exponential, with the rate calibrated per
  setting to give 30% censoring.

## Simulation settings

| Family | Setting | True parameters | Why |
|---|---|---|---|
| Weibull | W | scale 1, shape 1.5 | Reference: well-identified 2-parameter model |
| Burr XII | A | a 1, c 3, k 1 | Upside-down bathtub hazard |
| Burr XII | B | a 2, c 1.5, k 4 | Larger k, closer to the Weibull limit |
| ER-BurrXII | A | a 1, c 3, k 1.5 | Moderate tail |
| ER-BurrXII | B | a 5, c 10, k 0.5 | Region used in Anafo et al. (2021) Table 2 |
| GB2 | A | a 1, c 2, p 1.5, q 1.5 | Interior mode |
| GB2 | B | a 1, c 4, p 0.5, q 0.75 | Heavy tail, pole at 0 |

n ∈ {250, 1000, 4000}; censoring 0% and 30%; 200 replicates per cell.
Part B: n = 2000, 30% censoring, 100 replicates; log scale = intercept +
0.5·x1 − 0.3·x2, shapes constant; the fitted model lets every raw output depend
on x, so it must also learn zero slopes for the shapes.

Identifiability is read from three things: the median condition number of the
observed information (log scale), the largest correlation between log
estimates across replicates, and whether RMSE falls like n^−1/2.

## Results (rerun with the shape cap, 2026-10-06, 747 s on 15 cores)

Medians and RMSE are over converged fits only, so they flatter n = 250.

| Setting | Info. condition no. | Max \|corr\| of estimates | Failed fits n=250 (0% / 30% cens) | n ≥ 1000 |
|---|---|---|---|---|
| Weibull W | 1.4–2 | 0.1–0.4 | 0 / 0 | Bias < 1%, coverage 0.93–0.96 |
| Burr XII A | 70–98 | 0.95–0.97 | 0 / 0 | Bias < 1%, coverage 0.93–0.98 |
| Burr XII B | 450–970 | 0.99–1.00 | 3% / 7.5% | Bias ≤ 4%; small-n runaway toward Weibull limit |
| ER-BurrXII A | 330–1000 | 0.92–0.98 | 2% / 9.5% | Bias ≤ 6% |
| ER-BurrXII B (paper's region) | 1150–2500 | 0.92–0.99 | 9.5% / **26.5%** | Still 2–5.5% failures at n = 1000 |
| GB2 A | 1330–2500 | 0.92–0.98 | 14% / 17.5% | Bias ≤ 4%, coverage 0.95–0.99 |
| GB2 B | 855–1690 | 0.82–0.99 | 3% / 9.5% | Bias ≤ 3%, coverage 0.94–0.98 |

1. **Every head is correctly implemented and consistent.** At n ≥ 1000 the
   median relative bias is within a few percent, Wald coverage is close to
   0.95, and RMSE falls roughly like n^−1/2 (`results/recovery_rmse.png`).
2. **Identifiability, not the code, is the problem at small n.** The condition
   number does not shrink with n: it is a property of the parameterisation.
   Burr-type, ER and GB2 heads are 2–3 orders of magnitude worse conditioned
   than Weibull, and their log estimates are 0.95–0.99 correlated. At n = 250
   fits either fail (flat likelihood: gradient stalls near 5e-4, or a
   non-positive-definite information matrix) or run away along the ridge
   toward the Weibull limit (a, k → ∞ together; raw values up to 18–26).
3. **The ER constraint p = 1/c does not fix identifiability.** ER-BurrXII has
   three parameters but is conditioned as badly as four-parameter GB2, and
   setting B, from the 2021 paper, is the worst case in the study.
   (Before the shape cap, ER-BurrXII failures were higher: 5.5–33%.)
4. **Censoring roughly doubles** the condition number and the small-n failure
   rate.
5. **Covariates (Part B).** Weibull and Burr XII recover the scale slopes
   (bias ≤ 0.015) and keep spurious shape slopes at about 0 (RMSE ≤ 0.11).
   ER-BurrXII and GB2 recover the scale slopes (bias ≤ 0.04), but their
   spurious shape slopes are noisy (RMSE 0.15–0.33). A flexible network would
   fit noise into the shape outputs unless they are shared or penalised.

### Reparameterisation (objective 3; numbers from the rerun with the shape cap)

GB2 was refitted to the same datasets in two other parameterisations:
`gb2_genf` (generalized F: mu, sigma, Q, P; Cox 2008) and `gb2_lmom` (mean
and SD of log T, plus log p, log q). Maximum likelihood is invariant to
parameterisation, so these can only change how the optimiser behaves. Minimum
NLLs agreed to about 1e-5.

| GB2 setting, n = 250 | Condition no. (gb2 / genf / lmom) | Failed % (gb2 / genf / lmom) |
|---|---|---|
| A, 0% censored | 1639 / 506 / 909 | 14.0 / **0.0** / 0.5 |
| A, 30% censored | 2501 / 758 / 1327 | 17.5 / **0.5** / 0.5 |
| B, 0% censored | 1104 / 350 / 630 | 3.0 / 2.0 / 2.5 |
| B, 30% censored | 1693 / 533 / 925 | 9.5 / 5.5 / 5.5 |

Mean seconds per fit: gb2 1.5, genf 0.5, lmom 0.6. Before the cap the
reparameterised heads still failed 9–12% of the time at n = 250: many of
those failures were runs toward the numerically broken region that the cap
removes.

**Prediction is not affected by the identifiability problem**
(`results/survival_summary.csv`). The RMSE of fitted S(t) at the true
S = 0.5 / 0.1 / 0.01 times is about the same for every family and
parameterisation, and equal to Weibull fitted to Weibull data: for example,
at n = 250 with no censoring, about 0.03 / 0.015 / 0.005 for all of them.
Fits flagged as failed predict S(t) about as well as converged ones: at n = 250
the median absolute error at S = 0.5 is 0.016–0.028 for failed fits against
0.020 for converged ones. The single exception is one failed `gb2_lmom` fit
with an error of 0.44.

**Conclusions.**
1. Reparameterisation plus the shape cap fixes the optimiser: conditioning
   improves 2–3×, fits run about 3× faster, and small-n failures fall from
   14–17.5% to 0–5.5%. The flat direction is still there (shape estimates
   stay noisy), but fitting no longer breaks on it.
2. The flat direction barely changes the survival curve. The parameters are
   poorly identified at small n, but S(t), quantiles and tail probabilities
   are well identified. The identifiability problem matters for interpreting
   (and drawing inference on) per-patient shape and tail parameters, and for
   compute; it does not matter for prediction.
3. `gb2_lmom` is the best default head: the fastest, with the lowest estimate
   correlation at small n.

Not yet tested: whether GB2 heads beat Weibull when the truth is heavy-tailed
(misspecification). That is the real selling point and the next simulation.

**Earlier note on objective 3 (before the run above):** reparameterise toward quantities
the data inform nearly orthogonally: log median, upper-tail index c·q and
lower-tail index c·p. Keep shapes global and let only the scale or median
depend on x initially; then add penalties. Re-run this simulation to compare.

### Misspecification (run 2026-10-06, 1816 s, with the shape cap)

`sim_misspec.py`: 8 true distributions × 5 fitted heads (Weibull, log-normal,
Burr XII, ER-BurrXII, GB2 log-moment) plus "pick by AIC", n ∈ {250, 1000,
4000}, two censoring schemes (30% random; `admin15` = follow-up ends when 15%
are still alive, so S = 0.1 and below are extrapolations), 100 paired datasets
per cell. Outputs: `results/misspec_summary.csv`, `results/misspec_tail.png`.

**Worst case over the 8 true distributions** (lower is better):

| n, censoring | Metric | Weibull | Log-normal | Burr XII | ER-BurrXII | GB2 | AIC pick |
|---|---|---|---|---|---|---|---|
| 1000, random | excess log-loss | 0.315 | 0.215 | 0.034 | 0.132 | **0.005** | 0.006 |
| 1000, random | rel. error S at 1% | 1.00 | 8.90 | 1.32 | 1.11 | **0.47** | 0.49 |
| 1000, admin15 | rel. error S at 1% | 1.00 | 8.54 | 1.78 | 2.07 | 0.91 | **0.88** |
| 4000, random | rel. error S at 1% | 1.00 | 8.91 | 1.16 | 0.93 | 0.26 | **0.25** |
| 250, admin15 | excess log-loss | 0.544 | 0.218 | **0.058** | 0.324 | 1.137 | 0.205 |

1. **The two families used by Deep Survival Machines fail badly outside
   their family.** Weibull predicts S ≈ 0 at the 1% time for every
   heavy-tailed truth (relative error 1.0). Log-normal overestimates the 1%
   tail 6–9× for ER-BurrXII and Weibull truths.
2. **GB2 is never far from the best.** At n ≥ 1000 it has the best worst
   case on every metric. It doesn't beat the correct small family when that
   family is the truth (no free lunch), but it avoids every catastrophe.
3. **Extrapolation beyond follow-up is hard for every model.** With
   `admin15`, GB2's 1% tail error rises to 0.5–0.9. At n = 250 with
   administrative censoring, GB2 can be badly wrong (worst-case log-loss
   1.14); Burr XII is the safest there.
4. **AIC selection across heads is as good as GB2, or slightly better.** It
   picks the true family 64–93% of the time.
5. ER-BurrXII is a poor general-purpose head: it is good only when it is the
   truth.

Recovery rerun with the cap: small-n failures for `gb2_genf` and `gb2_lmom`
fell to 0–5.5% (from 11–12%). Plain GB2 is unchanged at 3–17.5%.

### Heads inside a neural network (run 2026-10-06, 426 s)

`network.py`: an MLP (64-64, ReLU, dropout 0.1) with any head on top. The
location output always depends on x; the shape outputs are either shared by
all patients ("global") or predicted per patient ("x"). Training: Adam,
mini-batches of 256, gradient clipping, early stopping on a 20% validation
split. `sim_network.py`: 5 covariates with nonlinear effects on the location;
truths Weibull AFT, log-normal, and GB2 whose tail is heavy for x1 > 0
(S ~ t^−1.5) and light otherwise; n ∈ {1000, 5000}; 30% censoring; 20 runs
per cell; scored against each test patient's true curve
(`results/network_summary.csv`, `results/network_results.png`).

Excess test log-loss, n = 5000 (median of 20 runs; lower is better):

| Head | Truth: Weibull AFT | Truth: log-normal | Truth: GB2, tail varies |
|---|---|---|---|
| Weibull | **0.031** | 0.130 | 9.5 |
| Log-normal | 0.116 | **0.026** | 0.185 |
| Burr XII (x) | 0.039 | 0.046 | **0.054** |
| GB2 | 0.034 | 0.035 | 0.123 |
| GB2 (x) | 0.037 | 0.036 | **0.055** |

1. **The heads train reliably inside a network.** There were no non-finite
   losses or skipped batches in 960 trainings.
2. **The robustness result carries over.** Weibull fails catastrophically
   when tails are heavy; log-normal is 4× worse on Weibull data. GB2 stays
   within about 1.3× of the best head everywhere.
3. **C-index cannot see any of this.** It is the same for every head
   (0.70–0.72, or 0.83 for the GB2 truth), so the paper must report log-loss
   and calibration.
4. **When the tail differs between patients, shapes must depend on x.** With
   shared shapes the heavy-tailed half is predicted to have almost no
   survivors (calibration at 1%: 0.06), and the light-tailed half 7× too
   many. With per-patient shapes this improves to 1.6 and 3.1. At n = 1000,
   per-patient shapes overfit when the true shape is constant.
5. **Burr XII is nearly as good as GB2 here,** and much cheaper (closed-form
   survival function; about 5× faster to train).
6. **Tail calibration at the 1% level is poor for every head** (predicted S
   1.3–4× too high at n = 5000, worse at n = 1000). Part of this is a
   Jensen effect: noise in the predicted location inflates the average of a
   convex tail probability.

### Real data, first run (superseded: network initialisation was not seeded; see the rerun below)

METABRIC, SUPPORT, GBSG and FLCHAIN (pycox), 5 × 5-fold cross-validation
(25 test folds), a stratified 20% of each training fold for early stopping,
the same MLP and training settings for every model, no tuning. 1,400 fits,
no errors. `results/real_raw.csv` and `results/real_summary.csv`.

Integrated Brier score (mean over 25 folds; lower is better):

| Model | METABRIC | GBSG | SUPPORT | FLCHAIN |
|---|---|---|---|---|
| GB2 (x) | **0.1712** | **0.1758** | 0.2111 | 0.0977 |
| Burr XII (x) | 0.1718 | 0.1762 | 0.2028 | 0.0983 |
| ER-BurrXII (x) | 0.1731 | 0.1776 | 0.2036 | 0.0980 |
| Log-normal (x) | 0.1732 | 0.1761 | 0.2026 | 0.0996 |
| Weibull (x) | 0.1732 | 0.1773 | 0.2033 | 0.0982 |
| DSM-style mixture, log-normal | 0.1721 | 0.1760 | **0.2024** | **0.0976** |
| DeepSurv | 0.1743 | 0.1774 | 0.2041 | 0.0981 |
| DeepHit | 0.1851 | 0.2018 | 0.2211 | 0.1130 |

1. Parametric heads with per-patient shapes beat DeepSurv on most datasets
   (GB2 lower IBS in 80–88% of folds on METABRIC, GBSG, FLCHAIN). DeepHit
   is worst everywhere despite the best METABRIC C-index (0.674), and fails
   D-calibration in every GBSG fold.
2. Among parametric heads and mixtures the differences are tiny
   (≤ 0.001 IBS; GB2 beats Burr XII or the mixture in only about half the
   folds).
3. GB2 trains badly in some folds: on SUPPORT it is worse than its own
   sub-models; on FLCHAIN one fold has test NLL 13.3 (median 2.94). GB2
   training needs a more robust start (for example, from a fitted Burr XII).
4. On SUPPORT, single-distribution heads fail D-calibration (0–8% of folds
   pass); mixtures pass 48–64%. SUPPORT mixes very early deaths with long
   survivors.
5. ER-BurrXII is competitive where event times peak at the start of
   follow-up (SUPPORT, FLCHAIN: it beats DeepSurv) and weaker where they peak
   later (METABRIC, peak around 35–47 months; GBSG, around 13–17 months),
   where it also fails D-calibration (GBSG: 8–16% of folds pass, against 100%
   for Burr XII). That is the expected consequence of its density always
   decreasing from t = 0.

Network simulation rerun with ER-BurrXII (n = 5000, excess log-loss): 0.05
on the Weibull truth (close to the best), 0.10 on log-normal (3× worse than
the best), 0.15–0.28 on the patient-varying GB2 tail (between log-normal
and Burr XII/GB2).

### Real data rerun: seeded, warm-started GB2, Burr XII mixture (2026-10-06, 285 s)

All 16 models, seeded initialisation, 1,600 fits, no errors. Added
`gb2_warm/x` (GB2 started from the fitted Burr XII network: p = 1, q = k,
same body; early stopping keeps the start as a candidate) and `mix_burr12/x`
(3-component Burr XII mixture).

Integrated Brier score (mean of 25 folds):

| Model | METABRIC | GBSG | SUPPORT | FLCHAIN |
|---|---|---|---|---|
| Mixture, Burr XII | **0.1713** | 0.1757 | 0.2020 | 0.0976 |
| Mixture, log-normal | 0.1716 | 0.1758 | **0.2019** | **0.0975** |
| Mixture, Weibull | 0.1722 | 0.1758 | 0.2021 | 0.0977 |
| GB2 (x), cold start | **0.1713** | **0.1755** | 0.2063 | 0.0976 |
| GB2 (x), warm start | 0.1721 | 0.1768 | 0.2029 | 0.0982 |
| Burr XII (x) | 0.1721 | 0.1768 | 0.2026 | 0.0984 |
| ER-BurrXII (x) | 0.1731 | 0.1774 | 0.2033 | 0.0980 |
| DeepSurv | 0.1743 | 0.1774 | 0.2041 | 0.0981 |
| DeepHit | 0.1851 | 0.2018 | 0.2211 | 0.1130 |

- **(a) Warm start.** It removes GB2's bad folds (worst test NLL on SUPPORT
  4.63 instead of 4.96; FLCHAIN 2.97 instead of 3.62) and never ends worse
  than its Burr XII start on validation. But it barely leaves that start
  (median validation-NLL gain 0.001), so its accuracy equals Burr XII's. The
  cold-started GB2 is slightly better on average on METABRIC and GBSG but
  has occasional bad folds. Robustness and flexibility trade off.
- **(b) Burr XII mixture.** Best or within 0.0002 of best on all four
  datasets; beats DeepSurv in 76–92% of folds; passes D-calibration on
  SUPPORT in 60% of folds (single heads: 0–12%). It ties the log-normal
  mixture on real data.
- **Network simulation with mixtures** (n = 5000, heavy tail varying by
  patient): excess log-loss is 0.064 for the Burr XII mixture, 0.090 for the
  log-normal mixture and 10.9 for the Weibull mixture (DSM's default
  component). In the heavy-tailed half, the log-normal mixture predicts 27%
  of the true 1% survivors. On log-normal and Weibull truths the Burr XII
  mixture is within 0.01 of the log-normal mixture.

**Headline for the paper:** a Burr XII mixture head matches the best deep
survival models on four clinical datasets and keeps its accuracy when tails
are heavy, where Weibull and log-normal heads and mixtures break down.

### MAPT-Burr XII head (2026-10-06)

`MAPTBurrXII` in `heads.py`: the modified alpha-power-transformed Burr XII of
Anafo, Ocloo & Nasiru (2024), with a scale a added; α = 1 is Burr XII, and the
formulas are written in log α so they are exact at and near α = 1. Tested
against the paper's eq. 5 (to 1e-10). In the far tail S_G / S_Burr → α² ln α /
(α − 1), so the tail index is Burr XII's. The paper's statement that the
hazard → ∞ is incorrect: it → 0, as for Burr XII.

Real data (IBS, 25 folds), MAPT-Burr XII (x) vs Burr XII (x): METABRIC
0.1719 vs 0.1721 (better in 68% of folds), GBSG 0.1766 vs 0.1768 (52%),
SUPPORT 0.2032 vs 0.2026 (44%), FLCHAIN 0.0980 vs 0.0984 (84%). D-calibration
on SUPPORT: 20% of folds pass vs 4%. Network simulation: within 0.004 of Burr
XII on every truth. In short, it is a small, consistent refinement of Burr XII
in the body of the distribution, and it ties the other top heads; it does not
reach the Burr XII mixture.

### Classical baselines and log-transformed features (2026-10-06, 3,898 s)

Added a linear Cox PH (lifelines; ridge penalty chosen from 0.001–1 by
validation log-likelihood, then refitted on train + validation) and a random
survival forest (scikit-survival; 200 trees, min leaf 15, sqrt features). The
first Cox run broke on 5 of 25 FLCHAIN folds (C-index 0.02–0.31): kappa and
lambda have extreme right tails, and the Cox linear predictor reached −1,057
for a few test patients. The fix, applied to every model: continuous features
with skewness > 2 and no negative values are log1p-transformed before
standardising (METABRIC x0; SUPPORT x12, x13; GBSG x4–x6; FLCHAIN kappa,
lambda, creatinine). All 20 models were rerun (2,000 fits, no errors). These
numbers supersede the earlier real-data tables (old file:
`results/real_raw_nolog.csv`).

IBS (mean of 25 folds), selected models:

| Model | METABRIC | GBSG | SUPPORT | FLCHAIN |
|---|---|---|---|---|
| Mixture, Burr XII | 0.1715 | 0.1752 | 0.2015 | **0.0974** |
| Mixture, log-normal | 0.1715 | **0.1750** | 0.2014 | 0.0975 |
| GB2 (x), cold start | **0.1714** | 0.1751 | 0.2026 | **0.0974** |
| Burr XII (x) | 0.1722 | 0.1762 | 0.2020 | 0.0980 |
| Random survival forest | 0.1729 | 0.1770 | **0.2003** | 0.0996 |
| DeepSurv | 0.1744 | 0.1777 | 0.2038 | 0.0975 |
| Cox PH (linear) | 0.1744 | 0.1772 | 0.2064 | 0.0976 |
| DeepHit | 0.1851 | 0.2017 | 0.2209 | 0.1131 |

Burr XII mixture beats Cox in 76% / 80% / 100% / 52% of folds (METABRIC, GBSG,
SUPPORT, FLCHAIN) and RSF in 68% / 88% / 24% / 100%. RSF is the best model on
SUPPORT; Cox ties the top on FLCHAIN. DeepSurv has the best calibration on
SUPPORT (D-calibration passes in 96% of folds).
