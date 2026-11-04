"""Second autocorrelation inequality: maximize L2^2 / (L1 * sup|autoconv| for 50 nonneg heights."""
import numpy as np
from scipy.optimize import minimize

N = 50
# interval [-1/4, 1/4], 50 equal sub-intervals
_h = np.linspace(-0.25, 0.25, N + 1)
_dx = _h[1] - _h[0]
_x = 0.5 * (_h[:-1] + _h[1:])          # midpoints of the 50 intervals


def _score(h):
    """Exact objective: L2^2 / (L1 * sup|autoconv|). Returns (value, valid)."""
    h = np.asarray(h, dtype=float)
    L1 = float(np.sum(h))
    if L1 <= 0.0:
        return 0.0, False
    ac = np.convolve(h, h, mode="full")
    sup = float(np.max(ac))
    if sup <= 0.0:
        return 0.0, False
    val = float(np.dot(h, h)) / (L1 * sup)
    return val, np.all(h >= -1e-12)


def _obj_and_grad(v, beta):
    """Smooth surrogate value + gradient (max kept sharp, L1 softened via softplus)."""
    h = np.maximum(v, 0.0)
    L2sq = float(np.dot(h, h))
    s = beta * h
    L1s = float(np.sum(np.logaddexp(0.0, s)))           # softplus sum
    ac = np.convolve(h, h, mode="full")
    sup = float(np.max(ac))
    if sup <= 0.0:
        return 0.0, np.zeros_like(v)
    val = L2sq / (L1s * sup)
    # d(val)/d(h)  (softplus derivative = sigmoid(beta*h); max derivative = 1 on support)
    ds = beta * np.exp(s) / (1.0 + np.exp(s))
    g = (2.0 * h * sup - L2sq * ds) / (L1s * sup)
    return val, g


def construct():
    import time
    t0 = time.time()
    DEADLINE = t0 + 105.0          # safety margin under 120s wall-clock
    rnd = np.random.default_rng(20240517)

    best = np.ones(N)
    best_v, _ = _score(best)

    def _try(h):
        nonlocal best, best_v
        h = np.asarray(h, dtype=float)
        v, ok = _score(h)
        if ok and v > best_v:
            best_v = v
            best = np.array(h, dtype=float)

    # ---- 1) global random-restart local search on the smooth surrogate ----
    beta = 4.0
    n_restarts = 30
    for r in range(n_restarts):
        if time.time() > DEADLINE:
            break
        if r == 0:
            v0 = np.ones(N)
        else:
            v0 = np.maximum(rnd.normal(0.0, 0.6, N), 0.0)
        res = minimize(
            _obj_and_grad, v0,
            args=(beta,),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 400, "maxls": 50},
        )
        if res.x is not None:
            _try(np.maximum(res.x, 0.0))

    # ---- 2) coarse-to-fine structured grid (peaked profiles) ----
    for grid in (4, 8, 16):
        if time.time() > DEADLINE:
            break
        for center in range(1, N - 1):
            for width in range(2, N):
                h = np.exp(-((np.arange(N) - center) ** 2) / (2.0 * width))
                for amp in (0.5, 1.0, 2.0, 4.0, 8.0):
                    if time.time() > DEADLINE:
                        break
                    _try(h * amp)
                # polish the best-amp profile for this (center, width)
                base = np.exp(-((np.arange(N) - center) ** 2) / (2.0 * width))
                res = minimize(
                    _obj_and_grad, base.copy(),
                    args=(beta,),
                    jac=True,
                    method="L-BFGS-B",
                    options={"maxiter": 200, "maxls": 30},
                )
                if res.x is not None:
                    _try(np.maximum(res.x, 0.0))

    # ---- 3) final local polish of the global best ----
    if time.time() < DEADLINE - 5.0:
        res = minimize(
            _obj_and_grad, best.copy(),
            args=(beta,),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 300, "maxls": 40},
        )
        if res.x is not None:
            _try(np.maximum(res.x, 0.0))

    return [max(0.0, float(x)) for x in best]
