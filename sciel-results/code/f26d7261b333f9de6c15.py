"""Chirp-based design: h[i] = cos(phi + k * i**2), optimized over (phi, k)."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 400
DEADLINE = 150.0


def _value(h):
    S = float(np.sum(h))
    if S == 0.0:
        return np.inf
    c = np.convolve(h, h)
    return abs(2.0 * N * float(np.max(c)) / (S * S))


def _h_from_params(p):
    phi, k = p
    i = np.arange(N, dtype=float)
    return np.cos(phi + k * i * i)


def _obj(p):
    return _value(_h_from_params(p))


def construct():
    t0 = time.time()
    best_p = (0.0, 0.0)
    best_v = _value(_h_from_params(best_p))

    # 1) Global search: differential evolution over (phi, k)
    bounds = [(-np.pi, np.pi), (-0.05, 0.05)]
    try:
        res = differential_evolution(
            _obj,
            bounds,
            seed=0,
            maxiter=40,
            tol=1e-8,
            popsize=12,
            init='sobol',
            polish=False,
        )
        if res.x is not None:
            v = _value(_h_from_params(res.x))
            if v < best_v:
                best_v = v
                best_p = tuple(res.x)
    except Exception:
        pass

    # 2) Local refinement: Nelder-Mead from the best DE point
    if time.time() - t0 < DEADLINE - 5:
        try:
            res2 = minimize(
                _obj,
                best_p,
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 400, 'maxfev': 600},
            )
            if res2.x is not None:
                v = _value(_h_from_params(res2.x))
                if v < best_v:
                    best_v = v
                    best_p = tuple(res2.x)
        except Exception:
            pass

    # 3) Safety: ensure we return before the deadline
    if time.time() - t0 > DEADLINE - 2:
        return np.ones(N)

    return _h_from_params(best_p)
