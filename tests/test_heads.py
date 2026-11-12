import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import integrate, special, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heads import GB2, HEADS, GB2GenF, GB2LogMoment, log_betainc  # noqa: E402

CASES = [
    ("weibull", dict(scale=1.0, shape=1.5)),
    ("weibull", dict(scale=3.0, shape=0.6)),
    ("lognormal", dict(mu=0.0, sigma=1.0)),
    ("lognormal", dict(mu=1.5, sigma=0.4)),
    ("burr12", dict(a=1.0, c=3.0, k=1.0)),
    ("burr12", dict(a=2.0, c=0.8, k=4.0)),
    ("maptb12", dict(a=1.0, c=4.02, k=2.86, alpha=4.05)),
    ("maptb12", dict(a=2.0, c=1.5, k=0.8, alpha=0.12)),
    ("maptb12", dict(a=1.0, c=3.0, k=1.0, alpha=1.0)),
    ("er_burr12", dict(a=1.0, c=3.0, k=1.5)),
    ("er_burr12", dict(a=5.0, c=10.0, k=0.5)),
    ("gb2", dict(a=1.0, c=2.0, p=1.5, q=1.5)),
    ("gb2", dict(a=1.0, c=4.0, p=0.5, q=0.75)),
    ("gb2_genf", GB2GenF.from_gb2(a=1.0, c=4.0, p=0.5, q=0.75)),
    ("gb2_genf", dict(mu=0.2, sigma=0.6, Q=-1.6, P=0.9)),
    ("gb2_lmom", GB2LogMoment.from_gb2(a=1.0, c=2.0, p=1.5, q=1.5)),
    ("gb2_lmom", dict(mu=-0.3, sigma=1.2, p=0.4, q=3.0)),
]
IDS = [f"{f}-{'-'.join(f'{v:g}' for v in p.values())}" for f, p in CASES]


def _fns(family, true):
    head = HEADS[family]()
    raw = torch.tensor(head.raw_from(**true), dtype=torch.float64)

    def pdf(t):
        return float(torch.exp(head.log_pdf(torch.tensor([t], dtype=torch.float64), raw)))

    def surv(t):
        return float(torch.exp(head.log_surv(torch.tensor([t], dtype=torch.float64), raw)))

    return head, raw, pdf, surv


@pytest.mark.parametrize("family,true", CASES, ids=IDS)
def test_density_integrates_to_one(family, true):
    _, _, pdf, _ = _fns(family, true)
    total = integrate.quad(pdf, 0, np.inf, limit=400)[0]
    assert total == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("family,true", CASES, ids=IDS)
def test_survival_is_tail_integral_of_density(family, true):
    _, _, pdf, surv = _fns(family, true)
    for t in (0.05, 0.5, 1.0, 3.0, 20.0):
        tail = integrate.quad(pdf, t, np.inf, limit=400)[0]
        assert surv(t) == pytest.approx(tail, rel=1e-6, abs=1e-12)


@pytest.mark.parametrize("family,true", CASES, ids=IDS)
def test_sampler_matches_cdf(family, true):
    head, _, _, surv = _fns(family, true)
    x = head.sample(np.random.default_rng(1), 4000, **true)
    p = stats.kstest(x, lambda t: 1 - np.array([surv(v) for v in np.atleast_1d(t)])).pvalue
    assert p > 0.001


def test_er_burr12_matches_paper_density():
    # Anafo et al. (2021) eq. 2.3: g(y) = (1+(y/a)^c)^(-k) / (k a B(1/c+1, k-1/c))
    a, c, k = 5.0, 10.0, 0.5
    _, _, pdf, _ = _fns("er_burr12", dict(a=a, c=c, k=k))
    mu = k * a * special.beta(1 / c + 1, k - 1 / c)
    for y in (0.5, 3.0, 5.0, 9.0):
        assert pdf(y) == pytest.approx((1 + (y / a) ** c) ** (-k) / mu, rel=1e-10)


@pytest.mark.parametrize("a,b", [(0.1, 0.1), (0.5, 3.0), (2.0, 0.3), (5.0, 5.0), (30.0, 0.7), (0.05, 40.0)])
def test_log_betainc_against_scipy(a, b):
    x = np.array([1e-8, 1e-3, 0.1, 0.4, 0.5, 0.6, 0.9, 0.999])
    xt = torch.tensor(x, dtype=torch.float64)
    got = log_betainc(torch.log(xt), torch.log1p(-xt),
                      torch.full_like(xt, a), torch.full_like(xt, b)).numpy()
    want = np.log(special.betainc(a, b, x))
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-9)


def test_log_betainc_gradients():
    def f(lx, a, b):
        return log_betainc(lx, torch.log1p(-torch.exp(lx)), a, b)

    for x in (0.2, 0.8):
        args = (torch.tensor([np.log(x)], dtype=torch.float64, requires_grad=True),
                torch.tensor([0.7], dtype=torch.float64, requires_grad=True),
                torch.tensor([2.5], dtype=torch.float64, requires_grad=True))
        assert torch.autograd.gradcheck(f, args)


@pytest.mark.parametrize("family", list(HEADS))
def test_batched_raw_and_gradients_finite(family):
    head = HEADS[family]()
    torch.manual_seed(0)
    raw = (0.5 * torch.randn(64, head.n_params, dtype=torch.float64)).requires_grad_()
    t = torch.exp(torch.randn(64, dtype=torch.float64) * 2)
    event = (torch.rand(64) < 0.6).double()
    loss = head.nll(raw, t, event)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(raw.grad).all()


@pytest.mark.parametrize("H", [GB2GenF, GB2LogMoment])
@pytest.mark.parametrize("gb", [dict(a=1.0, c=2.0, p=1.5, q=1.5), dict(a=3.0, c=0.7, p=4.0, q=0.3)])
def test_reparameterised_heads_equal_gb2(H, gb):
    t = torch.tensor([0.05, 0.7, 2.0, 30.0], dtype=torch.float64)
    ref = GB2()
    raw_ref = torch.tensor(ref.raw_from(**gb), dtype=torch.float64)
    head = H()
    raw = torch.tensor(head.raw_from(**H.from_gb2(**gb)), dtype=torch.float64)
    torch.testing.assert_close(head.log_pdf(t, raw), ref.log_pdf(t, raw_ref))
    torch.testing.assert_close(head.log_surv(t, raw), ref.log_surv(t, raw_ref))


@pytest.mark.parametrize("a,b", [(900.0, 950.0), (0.3, 990.0), (990.0, 2.0)])
def test_log_betainc_large_shapes(a, b):
    x = np.array([1e-6, 0.01, 0.3, 0.5, 0.7, 0.99])
    xt = torch.tensor(x, dtype=torch.float64)
    got = log_betainc(torch.log(xt), torch.log1p(-xt),
                      torch.full_like(xt, a), torch.full_like(xt, b)).numpy()
    want = np.log(special.betainc(a, b, x))
    finite = np.isfinite(want) & (want > -700)
    np.testing.assert_allclose(got[finite], want[finite], rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("family", ["gb2", "gb2_lmom", "gb2_genf", "er_burr12"])
def test_density_still_normalised_at_extreme_raw(family):
    # raw outputs far beyond any sensible value: shapes hit the cap, and the
    # density must still integrate to one (it reached 1e133 before the cap existed)
    head = HEADS[family]()
    raw = torch.tensor([0.0, 0.2, 60.0, 60.0][:head.n_params], dtype=torch.float64)
    if family == "gb2_genf":
        raw = torch.tensor([0.0, 0.2, 25.0, -60.0], dtype=torch.float64)

    # integrate the density of log T on a fine, wide grid: at the cap the
    # density is either a very narrow spike or spread over hundreds of log units
    u = torch.linspace(-650, 650, 4_000_001, dtype=torch.float64)
    total = float(torch.trapezoid(torch.exp(head.log_pdf(torch.exp(u), raw) + u), u))
    assert total == pytest.approx(1.0, abs=1e-6)


def test_genf_matches_documented_density():
    # flexsurv dgenf (Prentice 1975 / Cox 2008 parameterisation), coded independently
    def dgenf(x, mu, sigma, Q, P):
        d = np.sqrt(Q * Q + 2 * P)
        s1, s2 = 2 / (Q * Q + 2 * P + Q * d), 2 / (Q * Q + 2 * P - Q * d)
        w = (np.log(x) - mu) * d / sigma
        return d * (s1 / s2) ** s1 * np.exp(s1 * w) / (
            sigma * x * (1 + s1 * np.exp(w) / s2) ** (s1 + s2) * special.beta(s1, s2))

    head = GB2GenF()
    x = np.array([0.3, 1.0, 2.5])
    for Q in (0.37, -1.6, 0.0001):
        raw = torch.tensor(head.raw_from(0.2, 0.6, Q, 0.9), dtype=torch.float64)
        got = torch.exp(head.log_pdf(torch.tensor(x, dtype=torch.float64), raw)).numpy()
        np.testing.assert_allclose(got, dgenf(x, 0.2, 0.6, Q, 0.9), rtol=1e-10)


@pytest.mark.parametrize("family", ["mix_weibull", "mix_lognormal"])
def test_mixture_density_and_survival(family):
    head = HEADS[family]()
    raw = torch.tensor([0.3, -0.5, 0.0] + [-1.0, 0.2, 0.0, -0.3, 1.0, 0.5], dtype=torch.float64)

    def pdf(t):
        return float(torch.exp(head.log_pdf(torch.tensor([t], dtype=torch.float64), raw)))

    assert integrate.quad(pdf, 0, np.inf, limit=400)[0] == pytest.approx(1.0, abs=1e-6)
    for t in (0.1, 1.0, 4.0):
        s = float(torch.exp(head.log_surv(torch.tensor([t], dtype=torch.float64), raw)))
        assert s == pytest.approx(integrate.quad(pdf, t, np.inf, limit=400)[0], rel=1e-6)


def test_burr_mixture_density_and_survival():
    head = HEADS["mix_burr12"]()
    raw = torch.tensor([0.3, -0.5, 0.0] + [-1.0, 0.5, 0.2, 0.0, 1.0, -0.3, 1.0, 0.2, 0.4], dtype=torch.float64)

    def pdf(t):
        return float(torch.exp(head.log_pdf(torch.tensor([t], dtype=torch.float64), raw)))

    assert integrate.quad(pdf, 0, np.inf, limit=400)[0] == pytest.approx(1.0, abs=1e-6)
    for t in (0.1, 1.0, 4.0):
        s = float(torch.exp(head.log_surv(torch.tensor([t], dtype=torch.float64), raw)))
        assert s == pytest.approx(integrate.quad(pdf, t, np.inf, limit=400)[0], rel=1e-6)


def test_maptb12_is_burr_at_alpha_one_and_continuous_near_it():
    t = torch.tensor([0.05, 0.7, 2.0, 30.0], dtype=torch.float64)
    burr, mapt = HEADS["burr12"](), HEADS["maptb12"]()
    rb = torch.tensor(burr.raw_from(a=1.3, c=2.0, k=1.5), dtype=torch.float64)
    for la in (0.0, 1e-9, -1e-7, 1e-5):
        rm = torch.cat([rb, torch.tensor([la], dtype=torch.float64)])
        torch.testing.assert_close(mapt.log_pdf(t, rm), burr.log_pdf(t, rb), rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(mapt.log_surv(t, rm), burr.log_surv(t, rb), rtol=1e-4, atol=1e-5)


def test_maptb12_matches_paper_cdf():
    # Anafo, Ocloo & Nasiru (2024) eq. 5 with unit scale: G = (alpha^F - 1) / ((alpha - 1)(1 + alpha - alpha^F))
    alpha, xi, c = 4.05, 4.02, 2.86
    head = HEADS["maptb12"]()
    raw = torch.tensor(head.raw_from(a=1.0, c=xi, k=c, alpha=alpha), dtype=torch.float64)
    for x in (0.3, 0.9, 1.4):
        F = 1 - (1 + x ** xi) ** (-c)
        G = (alpha ** F - 1) / ((alpha - 1) * (1 + alpha - alpha ** F))
        s = float(torch.exp(head.log_surv(torch.tensor([x], dtype=torch.float64), raw)))
        assert s == pytest.approx(1 - G, rel=1e-10)
