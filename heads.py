"""Parametric likelihood heads for deep survival models.

Each head maps unconstrained network outputs ``raw`` (shape [n, n_params]) to
distribution parameters and gives the log density and log survival function
needed for the right-censored log-likelihood

    loglik_i = event_i * log f(t_i) + (1 - event_i) * log S(t_i).

Families
--------
Weibull(scale, shape)          S(t) = exp(-(t/scale)^shape)
LogNormal(mu, sigma)           log T ~ Normal(mu, sigma^2)
BurrXII(a, c, k)               S(t) = (1 + (t/a)^c)^(-k)
GB2(a, c, p, q)                f(t) = c t^(cp-1) / [a^(cp) B(p,q) (1+(t/a)^c)^(p+q)]
ERBurrXII(a, c, k)             f(t) = S_Burr(t) / E_Burr[T]  =  GB2(a, c, 1/c, k - 1/c)
MAPTBurrXII(a, c, k, alpha)    modified alpha-power-transformed Burr XII; alpha = 1 is Burr XII
GB2GenF(mu, sigma, Q, P)       GB2 in the generalized F parameterisation (Cox 2008)
GB2LogMoment(mu, sigma, p, q)  GB2 with mu, sigma = mean and SD of log T

Positive parameters are taken as exp(raw), so ``raw`` is the log parameter;
``log_raw`` marks the exceptions (mu and Q are used as is). GB2-type shape
parameters are smoothly capped below SHAPE_MAX (see below), so for them raw is
the log parameter only while the parameter is well below the cap.

ER-BurrXII needs k > 1/c (finite Burr mean); it is parameterised by
(log a, log c, log q) with q = k - 1/c, so every raw value is valid.

Everything runs in float64. The GB2 survival function is a regularized
incomplete beta function, which PyTorch lacks; ``log_betainc`` computes it by
a continued fraction written in torch, so gradients flow to x, a and b
through autograd.
"""


import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import special as sps

DTYPE = torch.float64
RAW_CLAMP = 30.0  # keeps exp(raw) finite when a network output runs away

BETACF_ITERS = 1000  # upper bound only; the loop stops at convergence

# GB2 shape parameters (p, q) are smoothly capped below SHAPE_MAX. Log-normal,
# Weibull and generalized gamma are limits of GB2 with p and/or q -> infinity,
# so on such data the likelihood pushes the shapes upward without bound; past
# ~1e3 the distribution is already indistinguishable from its limit, while the
# lgamma / continued-fraction arithmetic loses accuracy and the optimiser can
# exploit the error. For generalized F, P >= GENF_P_MIN keeps s1, s2 <~ 1e3.
SHAPE_MAX = 1e3
GENF_P_MIN = 2 / SHAPE_MAX


def lbeta(a, b):
    return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)


def _betacf(a, b, log_x, n_iter):
    """Continued fraction for I_x(a, b), modified Lentz (Numerical Recipes 6.4).

    Converges quickly for x < (a + 1) / (a + b + 2). Stops once every element
    has converged (at most n_iter steps); autograd differentiates through the
    iterations that ran.
    """
    tiny = 1e-300
    x = torch.exp(log_x)

    def fix(v):
        return torch.where(v.abs() < tiny, torch.full_like(v, tiny), v)

    qab, qap, qam = a + b, a + 1, a - 1
    c = torch.ones_like(x)
    d = 1 / fix(1 - qab * x / qap)
    h = d
    for m in range(1, n_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 / fix(1 + aa * d)
        c = fix(1 + aa / c)
        h = h * d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 / fix(1 + aa * d)
        c = fix(1 + aa / c)
        delta = d * c
        h = h * delta
        if (delta - 1).abs().max() < 1e-15:
            break
    return h


def _log_betainc_direct(log_x, log_1mx, a, b, n_iter):
    """log I_x(a, b) by continued fraction; valid for x < (a + 1) / (a + b + 2)."""
    return (a * log_x + b * log_1mx - lbeta(a, b) - torch.log(a)
            + torch.log(_betacf(a, b, log_x, n_iter)))


def log_betainc(log_x, log_1mx, a, b, n_iter=BETACF_ITERS):
    """log of the regularized incomplete beta function I_x(a, b).

    Takes log x and log(1 - x) separately so callers can pass both without
    cancellation. Below the switch point x* = (a + 1) / (a + b + 2) the
    continued fraction gives I directly; above it, I = 1 - I_{1-x}(b, a) and
    the subtracted term is at most about one half, so small values of I are
    never formed by cancellation. Each branch sees inputs clamped to its valid
    side, so the unused branch never produces NaN gradients through
    torch.where.
    """
    thr = (a + 1) / (a + b + 2)
    log_thr, log_1mthr = torch.log(thr), torch.log1p(-thr)
    direct = log_x < log_thr
    lo = _log_betainc_direct(torch.where(direct, log_x, log_thr),
                             torch.where(direct, log_1mx, log_1mthr), a, b, n_iter)
    comp = _log_betainc_direct(torch.where(direct, log_1mthr, log_1mx),
                               torch.where(direct, log_thr, log_x), b, a, n_iter)
    hi = torch.log1p(-torch.exp(torch.clamp(comp, max=-1e-16)))
    return torch.where(direct, lo, hi)


class Head:
    """Base class: subclasses define ``names``, ``params`` and the log f / log S."""

    names = ()

    @property
    def raw_names(self):
        """Name of the parameter each raw column is the log of."""
        return self.names

    @property
    def log_raw(self):
        """Whether each raw column is a log parameter (True) or used as is."""
        return (True,) * len(self.names)

    @property
    def n_params(self):
        return len(self.names)

    def params(self, raw):
        """dict of named positive parameters from raw [n, n_params]."""
        raise NotImplementedError

    def raw_from(self, **true):
        """raw vector (log scale) corresponding to named parameter values."""
        raise NotImplementedError

    def log_pdf(self, t, raw):
        raise NotImplementedError

    def log_surv(self, t, raw):
        raise NotImplementedError

    def sample(self, rng, n, **true):
        raise NotImplementedError

    def nll(self, raw, t, event):
        """Mean negative right-censored log-likelihood."""
        ll = torch.where(event.bool(), self.log_pdf(t, raw), self.log_surv(t, raw))
        return -ll.mean()

    @staticmethod
    def _exp(raw):
        return torch.exp(torch.clamp(raw, -RAW_CLAMP, RAW_CLAMP))

    @staticmethod
    def _shape(raw):
        """exp(raw) / (1 + exp(raw) / SHAPE_MAX): about exp(raw) when small, < SHAPE_MAX always."""
        raw = torch.clamp(raw, -RAW_CLAMP, RAW_CLAMP)
        return torch.exp(raw - F.softplus(raw - math.log(SHAPE_MAX)))

    @staticmethod
    def _shape_raw(x):
        """Inverse of _shape."""
        x = np.asarray(x, dtype=float)
        if np.any(x >= SHAPE_MAX):
            raise ValueError(f"shape parameter must be below SHAPE_MAX = {SHAPE_MAX:g}")
        return np.log(x) - np.log1p(-x / SHAPE_MAX)


class Weibull(Head):
    names = ("scale", "shape")

    def params(self, raw):
        e = self._exp(raw)
        return {"scale": e[..., 0], "shape": e[..., 1]}

    def raw_from(self, scale, shape):
        return np.log([scale, shape])

    def log_pdf(self, t, raw):
        p = self.params(raw)
        z = torch.log(t) - torch.log(p["scale"])
        return torch.log(p["shape"]) - torch.log(p["scale"]) + (p["shape"] - 1) * z - torch.exp(p["shape"] * z)

    def log_surv(self, t, raw):
        p = self.params(raw)
        return -torch.exp(p["shape"] * (torch.log(t) - torch.log(p["scale"])))

    def sample(self, rng, n, scale, shape):
        return scale * (-np.log(rng.uniform(size=n))) ** (1 / shape)


class LogNormal(Head):
    """log T ~ Normal(mu, sigma^2); raw = (mu, log sigma)."""

    names = ("mu", "sigma")
    log_raw = (False, True)

    def params(self, raw):
        return {"mu": raw[..., 0], "sigma": self._exp(raw)[..., 1]}

    def raw_from(self, mu, sigma):
        return np.array([mu, np.log(sigma)])

    def log_pdf(self, t, raw):
        p = self.params(raw)
        z = (torch.log(t) - p["mu"]) / p["sigma"]
        return -torch.log(t) - torch.log(p["sigma"]) - 0.5 * np.log(2 * np.pi) - 0.5 * z ** 2

    def log_surv(self, t, raw):
        p = self.params(raw)
        return torch.special.log_ndtr(-(torch.log(t) - p["mu"]) / p["sigma"])

    def sample(self, rng, n, mu, sigma):
        return np.exp(np.asarray(mu, float) + sigma * rng.normal(size=n))


class BurrXII(Head):
    names = ("a", "c", "k")

    def params(self, raw):
        e = self._exp(raw)
        return {"a": e[..., 0], "c": e[..., 1], "k": e[..., 2]}

    def raw_from(self, a, c, k):
        return np.log([a, c, k])

    def log_pdf(self, t, raw):
        p = self.params(raw)
        z = torch.log(t) - torch.log(p["a"])
        return (torch.log(p["c"]) + torch.log(p["k"]) - torch.log(p["a"])
                + (p["c"] - 1) * z - (p["k"] + 1) * F.softplus(p["c"] * z))

    def log_surv(self, t, raw):
        p = self.params(raw)
        return -p["k"] * F.softplus(p["c"] * (torch.log(t) - torch.log(p["a"])))

    def sample(self, rng, n, a, c, k):
        u = rng.uniform(size=n)
        return a * np.expm1(-np.log(u) / k) ** (1 / c)


class MAPTBurrXII(Head):
    """Modified alpha-power-transformed Burr XII (Anafo, Ocloo & Nasiru 2024), with a scale a.

    G(t) = (alpha^F - 1) / ((alpha - 1)(1 + alpha - alpha^F)), F the Burr XII(a, c, k) CDF.
    alpha = 1 is Burr XII itself (a removable 0/0), so everything is written in
    L = log alpha and s = S_Burr(t), with series-free stable pieces:

        S_G = alpha^2 (1 - alpha^-s) / ((alpha - 1) D),   D = 1 - alpha expm1(-s L) = 1 + alpha - alpha^F
        g   = alpha^(2 - s) (L / expm1(L)) f_Burr / D^2

    raw = (log a, log c, log k, log alpha); raw[3] = 0 is Burr XII. In the far
    tail S_G / S_Burr -> alpha^2 L / (alpha - 1): the tail index is Burr XII's.
    """

    names = ("a", "c", "k", "alpha")

    def __init__(self):
        self._burr = BurrXII()

    def params(self, raw):
        e = self._exp(raw)
        return {"a": e[..., 0], "c": e[..., 1], "k": e[..., 2], "alpha": e[..., 3]}

    def raw_from(self, a, c, k, alpha):
        return np.log([a, c, k, alpha])

    @staticmethod
    def _log_expm1_over(x):
        """log(expm1(x) / x), equal to x/2 + O(x^2) near 0; finite for all x."""
        small = x.abs() < 1e-6
        xs = torch.where(small, torch.ones_like(x), x)
        return torch.where(small, x / 2, torch.log(torch.expm1(xs) / xs))

    def _pieces(self, t, raw):
        L = torch.clamp(raw[..., 3], -RAW_CLAMP, RAW_CLAMP)
        log_s = self._burr.log_surv(t, raw[..., :3])
        s = torch.exp(log_s)
        log_d = torch.log1p(-torch.exp(L) * torch.expm1(-s * L))  # D = 1 + alpha - alpha^F > 0
        return L, log_s, s, log_d

    def log_surv(self, t, raw):
        L, log_s, s, log_d = self._pieces(t, raw)
        # (1 - alpha^-s) / (alpha - 1) = s * [(1 - e^-u)/u at u = sL] / [expm1(L)/L]
        return 2 * L + log_s + self._log_expm1_over(-s * L) - self._log_expm1_over(L) - log_d

    def log_pdf(self, t, raw):
        L, log_s, s, log_d = self._pieces(t, raw)
        return (2 - s) * L - self._log_expm1_over(L) + self._burr.log_pdf(t, raw[..., :3]) - 2 * log_d

    def sample(self, rng, n, a, c, k, alpha):
        p = rng.uniform(size=n)
        alpha = np.asarray(alpha, float)
        if np.all(np.abs(np.log(alpha)) < 1e-8):
            F = p
        else:  # invert G: alpha^F = (1 + p(alpha^2 - 1)) / (1 + p(alpha - 1))
            F = np.log((1 + p * (alpha ** 2 - 1)) / (1 + p * (alpha - 1))) / np.log(alpha)
        return a * np.expm1(-np.log1p(-np.clip(F, 0, 1 - 1e-16)) / k) ** (1 / c)


class GB2(Head):
    names = ("a", "c", "p", "q")

    def __init__(self, n_iter=BETACF_ITERS):
        self.n_iter = n_iter

    def params(self, raw):
        e = self._exp(raw)
        return {"a": e[..., 0], "c": e[..., 1], "p": self._shape(raw[..., 2]), "q": self._shape(raw[..., 3])}

    def raw_from(self, a, c, p, q):
        return np.array([np.log(a), np.log(c), self._shape_raw(p), self._shape_raw(q)])

    def _gb2(self, raw):
        """(a, c, p, q) used by the GB2 formulas; ER-BurrXII overrides this."""
        p = self.params(raw)
        return p["a"], p["c"], p["p"], p["q"]

    def log_pdf(self, t, raw):
        a, c, p, q = self._gb2(raw)
        z = torch.log(t) - torch.log(a)
        return torch.log(c) - torch.log(a) + (c * p - 1) * z - lbeta(p, q) - (p + q) * F.softplus(c * z)

    def log_surv(self, t, raw):
        # S(t) = I_u(q, p) with u = 1 / (1 + (t/a)^c)
        a, c, p, q = self._gb2(raw)
        cz = c * (torch.log(t) - torch.log(a))
        return log_betainc(-F.softplus(cz), -F.softplus(-cz), q, p, self.n_iter)

    @staticmethod
    def _sample_gb2(rng, n, a, c, p, q):
        x = rng.beta(p, q, size=n)
        return a * (x / (1 - x)) ** (1 / c)

    def sample(self, rng, n, a, c, p, q):
        return self._sample_gb2(rng, n, a, c, p, q)


class ERBurrXII(GB2):
    """Equilibrium-renewal Burr XII = GB2(a, c, 1/c, k - 1/c).

    raw = (log a, log c, log q) with q = k - 1/c > 0; ``params`` reports k.
    """

    names = ("a", "c", "k")
    raw_names = ("a", "c", "q")

    def params(self, raw):
        e = self._exp(raw)
        a, c, q = e[..., 0], e[..., 1], self._shape(raw[..., 2])
        return {"a": a, "c": c, "k": q + 1 / c}

    def raw_from(self, a, c, k):
        if k <= 1 / c:
            raise ValueError("ER-BurrXII needs k > 1/c (finite Burr XII mean)")
        return np.array([np.log(a), np.log(c), self._shape_raw(k - 1 / c)])

    def _gb2(self, raw):
        e = self._exp(raw)
        a, c, q = e[..., 0], e[..., 1], self._shape(raw[..., 2])
        return a, c, 1 / c, q

    def sample(self, rng, n, a, c, k):
        return self._sample_gb2(rng, n, a, c, 1 / c, k - 1 / c)


def _genf_shapes(Q, P, lib):
    """delta, s1, s2 of the generalized F, written to avoid cancellation.

    delta = sqrt(Q^2 + 2P), s1 = 2 / (delta (delta + Q)), s2 = 2 / (delta (delta - Q)),
    using delta +- Q = 2P / (delta -+ Q) on the side where the sum cancels.
    """
    delta = lib.sqrt(Q ** 2 + 2 * P)
    d_plus = lib.where(Q >= 0, delta + Q, 2 * P / (delta - Q))
    d_minus = lib.where(Q <= 0, delta - Q, 2 * P / (delta + Q))
    return delta, 2 / (delta * d_plus), 2 / (delta * d_minus)


class GB2GenF(GB2):
    """GB2 in the generalized F parameterisation (Prentice 1975; Cox 2008; flexsurv genf).

    log T = mu + sigma * w, w = log(F) / delta, F ~ F(2 s1, 2 s2). This is
    GB2(a, c, p, q) with p = s1, q = s2, c = delta / sigma and
    log a = mu + (sigma / delta) log(s2 / s1). Q is a skewness parameter and P
    a tail-weight parameter; P -> 0 gives the generalized gamma.
    raw = (mu, log sigma, Q, log P).
    """

    names = ("mu", "sigma", "Q", "P")
    log_raw = (False, True, False, True)

    def params(self, raw):
        e = self._exp(raw)
        return {"mu": raw[..., 0], "sigma": e[..., 1], "Q": torch.clamp(raw[..., 2], -RAW_CLAMP, RAW_CLAMP),
                "P": GENF_P_MIN + e[..., 3]}

    def raw_from(self, mu, sigma, Q, P):
        if P <= GENF_P_MIN:
            raise ValueError(f"P must exceed GENF_P_MIN = {GENF_P_MIN:g}")
        return np.array([mu, np.log(sigma), Q, np.log(P - GENF_P_MIN)])

    @staticmethod
    def from_gb2(a, c, p, q):
        """(mu, sigma, Q, P) for GB2(a, c, p, q)."""
        delta = np.sqrt(1 / p + 1 / q)
        sigma = delta / c
        return dict(mu=np.log(a) - np.log(q / p) / c, sigma=sigma,
                    Q=(1 / p - 1 / q) / delta, P=(delta ** 2 - ((1 / p - 1 / q) / delta) ** 2) / 2)

    @staticmethod
    def to_gb2(mu, sigma, Q, P, lib=np):
        delta, s1, s2 = _genf_shapes(Q, P, lib)
        c = delta / sigma
        return lib.exp(mu + lib.log(s2 / s1) / c), c, s1, s2

    def _gb2(self, raw):
        p = self.params(raw)
        return self.to_gb2(p["mu"], p["sigma"], p["Q"], p["P"], lib=torch)

    def sample(self, rng, n, mu, sigma, Q, P):
        return self._sample_gb2(rng, n, *self.to_gb2(np.asarray(mu, float), sigma, np.float64(Q), P))


class GB2LogMoment(GB2):
    """GB2 parameterised by the mean and SD of log T plus the beta shapes.

    log T = log a + logit(X) / c with X ~ Beta(p, q), so
    mu = E[log T] = log a + (digamma(p) - digamma(q)) / c and
    sigma = SD[log T] = sqrt(trigamma(p) + trigamma(q)) / c.
    raw = (mu, log sigma, log p, log q).
    """

    names = ("mu", "sigma", "p", "q")
    log_raw = (False, True, True, True)

    def params(self, raw):
        e = self._exp(raw)
        return {"mu": raw[..., 0], "sigma": e[..., 1], "p": self._shape(raw[..., 2]), "q": self._shape(raw[..., 3])}

    def raw_from(self, mu, sigma, p, q):
        return np.array([mu, np.log(sigma), self._shape_raw(p), self._shape_raw(q)])

    @staticmethod
    def from_gb2(a, c, p, q):
        return dict(mu=np.log(a) + (sps.digamma(p) - sps.digamma(q)) / c,
                    sigma=np.sqrt(sps.polygamma(1, p) + sps.polygamma(1, q)) / c, p=p, q=q)

    def _gb2(self, raw):
        prm = self.params(raw)
        p, q = prm["p"], prm["q"]
        c = torch.sqrt(torch.polygamma(1, p) + torch.polygamma(1, q)) / prm["sigma"]
        a = torch.exp(prm["mu"] - (torch.digamma(p) - torch.digamma(q)) / c)
        return a, c, p, q

    def sample(self, rng, n, mu, sigma, p, q):
        c = np.sqrt(sps.polygamma(1, p) + sps.polygamma(1, q)) / sigma
        a = np.exp(np.asarray(mu, float) - (sps.digamma(p) - sps.digamma(q)) / c)
        return self._sample_gb2(rng, n, a, c, p, q)


class Mixture(Head):
    """K-component mixture of a base head, as in Deep Survival Machines (Nagpal et al. 2021).

    raw = (K mixture logits, then the K blocks of base raw parameters). In a
    network every raw output depends on x, so both the weights and the
    components vary by patient.
    """

    def __init__(self, base, k=3):
        self.base, self.k = HEADS[base](), k
        self.names = tuple(f"w{j}" for j in range(k)) + tuple(
            f"{nm}{j}" for j in range(k) for nm in self.base.names)

    @property
    def log_raw(self):
        return (False,) * self.k + tuple(lr for _ in range(self.k) for lr in self.base.log_raw)

    def _split(self, raw):
        logw = torch.log_softmax(raw[..., :self.k], dim=-1)
        comp = raw[..., self.k:].reshape(*raw.shape[:-1], self.k, self.base.n_params)
        return logw, comp

    def params(self, raw):
        logw, comp = self._split(raw)
        out = {f"w{j}": torch.exp(logw[..., j]) for j in range(self.k)}
        for j in range(self.k):
            for nm, v in self.base.params(comp[..., j, :]).items():
                out[f"{nm}{j}"] = v
        return out

    def log_pdf(self, t, raw):
        logw, comp = self._split(raw)
        return torch.logsumexp(logw + self.base.log_pdf(t.unsqueeze(-1), comp), dim=-1)

    def log_surv(self, t, raw):
        logw, comp = self._split(raw)
        return torch.logsumexp(logw + self.base.log_surv(t.unsqueeze(-1), comp), dim=-1)

    def init_raw(self, t):
        """Equal weights; component locations spread around the median observed time."""
        raw = np.zeros(self.n_params)
        for j, offset in enumerate(np.linspace(-0.75, 0.75, self.k)):
            raw[self.k + j * self.base.n_params] = np.log(np.median(t)) + offset
        return raw

    def sample(self, rng, n, **true):
        raise NotImplementedError("Mixture is only fitted, not simulated from")


HEADS = {"weibull": Weibull, "lognormal": LogNormal, "burr12": BurrXII, "er_burr12": ERBurrXII, "gb2": GB2,
         "gb2_genf": GB2GenF, "gb2_lmom": GB2LogMoment,
         "mix_weibull": lambda: Mixture("weibull"), "mix_lognormal": lambda: Mixture("lognormal"),
         "mix_burr12": lambda: Mixture("burr12"), "maptb12": MAPTBurrXII}
