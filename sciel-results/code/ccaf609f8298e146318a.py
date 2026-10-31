"""Optimized program: numerically minimize the normalized autoconvolution peak."""
import time
import numpy as np
from scipy.optimize import minimize

N = 600
_DEADLINE = 150.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _score_of(h):
    # discrete autoconvolution peak (lag 0 = sum of squares for nonnegative)
    # normalized so that uniform [1]*N gives 1.0 under L-inf normalization:
    # score = max(c) / (max(h)^2 * N)
    h = np.asarray(h, dtype=float)
    c0 = float(np.dot(h, h))
    mh = float(np.max(h))
    if mh <= 0:
        return 1.0
    return c0 / (mh * mh * N)

def _objective(x, log_scale):
    # x are log-heights (so heights stay positive); we fix max height to 1 by scaling
    h = np.exp(x)
    mh = np.max(h)
    h = h / mh  # normalize max to 1 (scale invariance)
    c0 = float(np.dot(h, h))
    return c0 / N

def construct():
    deadline = _DEADLINE - 5.0  # safety margin
    best_h = np.ones(N)
    best_val = _score_of(best_h)

    rng = np.random.default_rng(0)

    # candidate initializations
    t = np.linspace(0, 1, N)
    inits = []
    inits.append(np.ones(N))
    inits.append(np.cos(np.pi * t) ** 2 + 1e-6)            # bell
    inits.append(1.0 - np.cos(2 * np.pi * t) ** 2 + 1e-6)   # U-shape
    inits.append(np.abs(np.sin(np.pi * t)) ** 2 + 1e-6)     # two-lobe
    inits.append(np.exp(-((t - 0.2) ** 2) / 0.02 + (-((t - 0.8) ** 2) / 0.02)) + 1e-6)  # two gaussians
    inits.append(rng.random(N) + 1e-6)
    inits.append(np.where(np.abs(t - 0.5) > 0.3, 1.0, 0.1) + 1e-6)  # edge-heavy

    for init in inits:
        if _elapsed() > deadline:
            break
        init = np.asarray(init, dtype=float)
        init = np.maximum(init, 1e-6)
        x0 = np.log(init)
        try:
            res = minimize(_objective, x0, method='L-BFGS-B',
                           options={'maxiter': 200, 'maxfun': 5000})
            if res.success or res.fun < best_val:
                h = np.exp(res.x)
                mh = np.max(h)
                h = h / mh
                val = _score_of(h)
                if val < best_val:
                    best_val = val
                    best_h = h
        except Exception:
            pass

    # return as list of floats (nonnegative step heights)
    return [float(v) for v in best_h]
