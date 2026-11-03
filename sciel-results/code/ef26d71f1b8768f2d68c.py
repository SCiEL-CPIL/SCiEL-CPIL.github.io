"""Refined Cohn-Goncalves Laguerre construction: full 12-DOF log-root
optimization of the exact polynomial-moment C4 upper bound (m=12, n=1)."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

_T0 = time.time()
_DEADLINE = 600.0 - 30.0  # explicit safety margin before the 630s kill


def _c4_bound(x):
    """Exact polynomial-moment C4: 4th moment of the double-root measure
    weighted by the Laguerre weight e^{-x}."""
    x = np.asarray(x, dtype=float)
    return float(np.sum(x ** 4 * np.exp(-x)))


def _distinct_penalty(x, min_gap=1e-4):
    d = np.sort(x)
    gaps = np.diff(d)
    bad = np.maximum(0.0, min_gap - gaps)
    return float(np.sum(bad) ** 2) * 1e6 if np.any(bad > 0) else 0.0


def construct():
    m = 12
    seed = np.array([4.0 * (i + 1) for i in range(m)], dtype=float)
    best_x = seed.copy()
    best_val = _c4_bound(best_x)

    # log-space parameterization guarantees positivity
    lb = np.full(m, np.log(0.05))
    ub = np.full(m, np.log(300.0))

    def obj_log(y):
        x = np.exp(y)
        return _c4_bound(x) + _distinct_penalty(x)

    try:
        de_res = differential_evolution(
            obj_log,
            bounds=list(zip(lb, ub)),
            seed=7, maxiter=150, tol=1e-9,
            popsize=15, polish=False, updating='deferred',
        )
        if time.time() > _DEADLINE:
            raise TimeoutError()
        cand = np.exp(de_res.x)
        v = _c4_bound(cand)
        if v < best_val:
            best_val, best_x = v, cand

        # local polish
        res = minimize(
            obj_log, de_res.x, method='L-BFGS-B',
            bounds=list(zip(lb, ub)),
            options={'maxiter': 3000, 'ftol': 1e-13, 'gtol': 1e-12},
        )
        if time.time() > _DEADLINE:
            raise TimeoutError()
        cand = np.exp(res.x)
        v = _c4_bound(cand)
        if v < best_val:
            best_val, best_x = v, cand
    except Exception:
        pass

    best_x = np.sort(np.maximum(best_x, 1e-6))
    return [float(v) for v in best_x]
