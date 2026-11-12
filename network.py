"""Neural survival model: an MLP backbone with a parametric likelihood head.

The network maps covariates x to the head's raw parameters. The first raw
output (log scale, log a or mu) always depends on x. The remaining "shape"
outputs either depend on x too (``shapes="x"``) or are single parameters
shared by everyone (``shapes="global"``), which the earlier covariate
simulation suggested is safer for GB2-type heads.

Training: Adam on the mean censored NLL, mini-batches, gradient clipping,
early stopping on a validation split, best weights restored.
"""

import copy

import numpy as np
import torch
from torch import nn

from heads import DTYPE, HEADS


class SurvivalNet(nn.Module):
    def __init__(self, n_features, family, hidden=(64, 64), dropout=0.1, shapes="global"):
        super().__init__()
        if shapes not in ("global", "x"):
            raise ValueError("shapes must be 'global' or 'x'")
        self.head = HEADS[family]()
        self.shapes = shapes
        P = self.head.n_params
        layers, width = [], n_features
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        self.body = nn.Sequential(*layers)
        n_out = P if shapes == "x" else 1
        self.out = nn.Linear(width, n_out)
        # start close to an intercept-only model: small output weights
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        self.shape_raw = nn.Parameter(torch.zeros(P - 1)) if shapes == "global" else None
        self.to(DTYPE)

    def init_intercept(self, raw0):
        """Set the output bias so that every subject starts at raw0 (length n_params)."""
        raw0 = torch.as_tensor(raw0, dtype=DTYPE)
        with torch.no_grad():
            if self.shapes == "x":
                self.out.bias.copy_(raw0)
            else:
                self.out.bias.copy_(raw0[:1])
                self.shape_raw.copy_(raw0[1:])

    def forward(self, x):
        o = self.out(self.body(x))
        if self.shapes == "x":
            return o
        return torch.cat([o, self.shape_raw.expand(len(x), -1)], dim=1)

    def nll(self, x, t, e):
        return self.head.nll(self(x), t, e)


def train(model, x, t, e, val_frac=0.2, lr=1e-3, weight_decay=1e-4, batch=256, max_epochs=400,
          patience=20, seed=0, raw0=None, val=None):
    """Fit in place; returns a dict with the training history summary.

    ``val`` = (x, t, e) uses that validation set for early stopping and trains
    on all of (x, t, e); otherwise a random ``val_frac`` of (x, t, e) is held out.
    """
    g = torch.Generator().manual_seed(seed)
    if val is not None:
        xv, tv, ev = val
        xt, tt, et = x, t, e
    else:
        perm = torch.randperm(len(t), generator=g)
        n_val = max(int(val_frac * len(t)), 1)
        iv, it = perm[:n_val], perm[n_val:]
        xv, tv, ev = x[iv], t[iv], e[iv]
        xt, tt, et = x[it], t[it], e[it]
    if raw0 is not None:
        model.init_intercept(raw0)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # the starting point is itself a candidate, so a warm start can never end
    # worse on validation than where it began
    model.eval()
    with torch.no_grad():
        v0 = float(model.nll(xv, tv, ev))
    best = v0 if np.isfinite(v0) else np.inf
    best_state, bad, epoch, skipped = copy.deepcopy(model.state_dict()), 0, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(tt), generator=g)
        for k in range(0, len(tt), batch):
            idx = order[k:k + batch]
            loss = model.nll(xt[idx], tt[idx], et[idx])
            if not torch.isfinite(loss):
                skipped += 1
                continue
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(model.nll(xv, tv, ev))
        if np.isfinite(v) and v < best - 1e-5:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return dict(epochs=epoch, best_val_nll=best, skipped_batches=skipped)


@torch.no_grad()
def predict_raw(model, x):
    model.eval()
    return model(x)


def survival_quantile(head, raw, s, lo=-40.0, hi=40.0, iters=80):
    """Per-subject time where S(t) = s, by bisection on log t (S is decreasing)."""
    lo = torch.full((len(raw),), lo, dtype=DTYPE)
    hi = torch.full((len(raw),), hi, dtype=DTYPE)
    target = np.log(s)
    with torch.no_grad():
        for _ in range(iters):
            mid = (lo + hi) / 2
            above = head.log_surv(torch.exp(mid), raw) > target
            lo = torch.where(above, mid, lo)
            hi = torch.where(above, hi, mid)
    return torch.exp((lo + hi) / 2)
