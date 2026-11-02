"""Hermite-Gaussian tuning for the L4 uncertainty constant C4.

We minimize the test-function upper bound
    B(c) = ||g||_4^4 / ||g||_2^4
for g(x) = (c0 H0 + c4 H4 + c8 H8 + c12 H12) e^{-x^2/2},
with c12 fixed by g(0)=0.  Lower B is better.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import eval_hermite
from scipy.optimize import minimize, differential_evolution

_DEADLINE = 180.0 - 25.0  # safety margin below the 210s kill / 180s limit

# Precompute a fine integration grid once.
def _build_grid(n=40001, half=12.0):
    x = np.linspace(-half, half, n)
    # weights for trapezoidal
    w = np.empty(n)
    w[1:-1] = x[2:] - x[:-2]
    w[0] = x[1] - x[0]
    w[-1] = x[-1] - x[-2]
    return x, w

_X, _W = _build_grid()
_GAUSS = np.exp(-0.5 * _X * _X)
# Hermite values on the grid
_H = np.stack([eval_hermite(n, _X) for n in (0, 4, 8, 12)], axis=1)  # (N,4)
_H00 = np.array([eval_hermite(0, 0.0), eval_hermite(4, 0.0),
                 eval_hermite(8, 0.0), eval_hermite(12, 0.0)])

def _coeffs12(c0, c4, c8):
    # c12 such that g(0)=0
    num = c0 * _H00[0] + c4 * _H00[1] + c8 * _H00[2]
    return -num / _H00[3]

def _objective(c, return_full=False):
    c0, c4, c8 = c
    c12 = _coeffs12(c0, c4, c8)
    poly = c0 * _H[:, 0] + c4 * _H[:, 1] + c8 * _H[:, 2] + c12 * _H[:, 3]
    g2 = poly * poly          # g^2 / e^{-x^2} factor handled below
    # ||g||_2^2 = ∫ poly^2 e^{-x^2} dx
    n2 = np.dot(_W * _GAUSS * _GAUSS, g2)
    # ||g||_4^4 = ∫ poly^4 e^{-2x^2} dx
    n4 = np.dot(_W * _GAUSS * _GAUSS * _GAUSS * _GAUSS, g2 * g2)
    if n2 <= 0:
        return 1e300
    B = n4 / (n2 * n2)
    return B

def _safe_objective(c):
    try:
        v = _objective(c)
        if not np.isfinite(v):
            return 1e300
        return v
    except Exception:
        return 1e300

def construct():
    t_start = time.time()
    best_c = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_v = _safe_objective(best_c)

    def remaining():
        return _DEADLINE - (time.time() - t_start)

    # 1) Global search with differential evolution (bounded, robust).
    if remaining() > 20:
        try:
            bounds = [(-3.0, 3.0), (-1.0, 1.0), (-0.1, 0.1)]
            res = differential_evolution(
                _safe_objective, bounds,
                seed=1234, maxiter=60, popsize=12,
                tol=1e-8, polish=False,
                init='sobol',
                updating='immediate',
                workers=1,
            )
            if res.x is not None and np.isfinite(res.fun) and res.fun < best_v:
                best_v = float(res.fun)
                best_c = np.asarray(res.x, dtype=float)
        except Exception:
            pass

    # 2) Local polish from several starts (L-BFGS-B), each guarded.
    starts = [best_c.copy(),
              np.array([-113 / 100, 1 / 25, 1 / 3240]),
              np.array([-1.0, 0.04, 0.0003]),
              np.array([-1.2, 0.05, 0.0002]),
              np.array([-0.9, 0.03, 0.0004])]
    for s in starts:
        if remaining() < 5:
            break
        try:
            r = minimize(_safe_objective, s, method='L-BFGS-B',
                         jac=_grad, options={'maxiter': 200, 'gtol': 1e-12})
            if r.x is not None and np.isfinite(r.fun) and r.fun < best_v:
                best_v = float(r.fun)
                best_c = np.asarray(r.x, dtype=float)
        except Exception:
            pass

    # 3) Final exact high-precision polish (Nelder-Mead tiny steps).
    if remaining() > 3:
        try:
            r = minimize(_safe_objective, best_c, method='Nelder-Mead',
                         options={'xatol': 1e-14, 'fatol': 1e-14,
                                  'maxiter': 2000, 'adaptive': True})
            if r.x is not None and np.isfinite(r.fun) and r.fun < best_v:
                best_v = float(r.fun)
                best_c = np.asarray(r.x, dtype=float)
        except Exception:
            pass

    return np.array([best_c[0], best_c[1], best_c[2]])

def _grad(c):
    h = 1e-7
    g = np.empty(3)
    base = _safe_objective(c)
    for i in range(3):
        cp = c.copy(); cp[i] += h
        cm = c.copy(); cm[i] -= h
        g[i] = (_safe_objective(cp) - _safe_objective(cm)) / (2 * h)
    return g
# EVOLVE-BLOCK-END
