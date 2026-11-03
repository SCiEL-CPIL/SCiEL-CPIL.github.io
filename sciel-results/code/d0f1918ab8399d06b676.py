"""Rewritten program: stronger optimization with cached analytic evaluation and a more thorough multi-stage search."""

import numpy as np
from scipy.optimize import differential_evolution, minimize, basinhopping
import time
from math import factorial, sqrt, pi

_START = time.time()
_DEADLINE = _START + 172.0


def _time_left():
    return _DEADLINE - time.time()


# ---------------------------------------------------------------------------
# Analytic pieces for the Hermite-Gaussian uncertainty product
# ---------------------------------------------------------------------------

indices = np.array([0, 4, 8, 12], dtype=int)

# Norms: <H_n, H_n> = sqrt(pi) * 2^n * n!
_norms_full = np.array([sqrt(pi) * (2.0 ** n) * float(factorial(n)) for n in indices])

# <x^2 H_n, H_n> = (n + 0.5) * <H_n, H_n>
_x2_diag = (indices + 0.5) * _norms_full

# H_n(0) for even n: (-1)^{n/2} * n! / (n/2)!
def _H_at_0(n):
    k = n // 2
    val = float(factorial(n)) / float(factorial(k))
    if k % 2:
        val = -val
    return val

_H0 = np.array([_H_at_0(n) for n in indices])

# Odd indices that appear in Q(x) = P'(x) - x P(x)
_odd = np.array([1, 3, 5, 7, 9, 11, 13], dtype=int)
_odd_norms = np.array([sqrt(pi) * (2.0 ** k) * float(factorial(k)) for k in _odd])

# Precompute the matrix M (4x7) such that d = M @ coeffs, where
# coeffs = [c0, c4, c8, c12] and d = [d1, d3, d5, d7, d9, d11, d13]
#
# From the recurrence:
#   P' terms:  c_n * n * H_{n-1}
#   -x P terms: c_n * (n/2) * H_{n-1}   (since x H_n = n H_{n-1} + (n+1) H_{n+1})
# Wait: x H_n = n H_{n-1} + (n+1) H_{n+1}, so -x P = -c_n [n H_{n-1} + (n+1) H_{n+1}]
# P' = c_n * n * H_{n-1}
# So Q = P' - xP = c_n [n H_{n-1} - n H_{n-1} - (n+1) H_{n+1}] = -c_n (n+1) H_{n+1}
#
# Hmm, that gives only H_{n+1} terms, which for even n gives odd indices n+1.
# Let me re-derive carefully.
#
# Standard recurrence: x H_n(x) = n H_{n-1}(x) + (n+1) H_{n+1}(x)
# Differentiation: H_n'(x) = 2n H_{n-1}(x)
#
# So Q(x) = P'(x) - x P(x)
# P'(x) = sum_n c_n * 2n * H_{n-1}(x)
# x P(x) = sum_n c_n * [n H_{n-1}(x) + (n+1) H_{n+1}(x)]
#
# Q(x) = sum_n c_n [2n H_{n-1} - n H_{n-1} - (n+1) H_{n+1}]
#      = sum_n c_n [n H_{n-1} - (n+1) H_{n+1}]
#
# For n=0:  c0 [0*H_{-1} - 1*H_1] = -c0 * H_1
# For n=4:  c4 [4*H_3 - 5*H_5]
# For n=8:  c8 [8*H_7 - 9*H_9]
# For n=12: c12 [12*H_11 - 13*H_13]
#
# So d1 = -c0, d3 = 4*c4, d5 = -5*c4, d7 = 8*c8, d9 = -9*c8, d11 = 12*c12, d13 = -13*c12

_M = np.zeros((4, 7))
# Row 0: c0
_M[0, 0] = -1.0
# Row 1: c4
_M[1, 1] = 4.0
_M[1, 2] = -5.0
# Row 2: c8
_M[2, 3] = 8.0
_M[2, 4] = -9.0
# Row 3: c12
_M[3, 5] = 12.0
_M[3, 6] = -13.0


def _bound_from_coeffs(coeffs):
    """Given 4 coefficients [c0,c4,c8,c12], return the uncertainty product sqrt(<x^2><p^2>)."""
    norm_sq = float(coeffs @ (_norms_full * coeffs))
    if norm_sq < 1e-30:
        return 1e30
    x2_num = float(coeffs @ (_x2_diag * coeffs))
    d = _M @ coeffs
    p2_num = float(d @ (_odd_norms * d))
    dx2 = x2_num / norm_sq
    dp2 = p2_num / norm_sq
    if dx2 < 0 or dp2 < 0:
        return 1e30
    return np.sqrt(dx2 * dp2)


def _objective(c3):
    """Objective taking 3 free coefficients [c0, c4, c8]; c12 is derived from g(0)=0."""
    c0, c4, c8 = c3
    c12 = -(c0 * _H0[0] + c4 * _H0[1] + c8 * _H0[2]) / _H0[3]
    coeffs = np.array([c0, c4, c8, c12])
    return _bound_from_coeffs(coeffs)


# Seed
_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
_SEED_VAL = _objective(_SEED)


def _polish(x0, budget=30.0):
    """Run a sequence of local optimizations from x0, returning the best result."""
    best_x = np.array(x0, dtype=float)
    best_v = _objective(best_x)

    methods = [
        ('Nelder-Mead', {'maxiter': 20000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True, 'initial_simplex': None}),
        ('Powell', {'maxiter': 20000, 'xtol': 1e-14, 'ftol': 1e-16}),
        ('L-BFGS-B', {'maxiter': 50000, 'gtol': 1e-14, 'ftol': 1e-15}),
    ]

    for method_name, opts in methods:
        if _time_left() < 5:
            break
        try:
            res = minimize(_objective, best_x, method=method_name, options=opts)
            if res.fun < best_v:
                best_v = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    return best_x, best_v


def _basin_hopping_run(x0, n_iter=50):
    """Run basin hopping from x0."""
    if _time_left() < 20:
        return None, None
    best_x = np.array(x0, dtype=float)
    best_v = _objective(best_x)
    try:
        res = basinhopping(_objective, best_x, niter=n_iter, minimizer_kwargs={'method': 'Nelder-Mead',
                                                                                 'options': {'xatol': 1e-12,
                                                                                             'fatol': 1e-14,
                                                                                             'adaptive': True,
                                                                                             'maxiter': 5000}})
        if res.fun < best_v:
            best_v = res.fun
            best_x = res.x.copy()
    except Exception:
        pass
    return best_x, best_v


def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 172.0

    best_x = _SEED.copy()
    best_v = _SEED_VAL

    # ------------------------------------------------------------------
    # Stage 1: Broad differential evolution
    # ------------------------------------------------------------------
    if _time_left() > 50:
        try:
            # Wide bounds
            bounds_wide = [(-20.0, 20.0), (-20.0, 20.0), (-10.0, 10.0)]
            res = differential_evolution(
                _objective, bounds_wide, seed=42, maxiter=500, popsize=40,
                tol=1e-12, polish=False, init='sobol', mutation=(0.5, 1.0),
                recombination=0.7
            )
            if res.fun < best_v:
                best_v = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Stage 2: Local polish from the DE result and the seed
    # ------------------------------------------------------------------
    for x0 in [best_x, _SEED]:
        if _time_left() < 10:
            break
        px, pv = _polish(x0)
        if pv < best_v:
            best_v = pv
            best_x = px

    # ------------------------------------------------------------------
    # Stage 3: Additional DE runs with tighter bounds around the best
    # ------------------------------------------------------------------
    if _time_left() > 40:
        try:
            spread = max(1.0, 0.5 * np.max(np.abs(best_x)))
            bounds_tight = [
                (best_x[0] - spread, best_x[0] + spread),
                (best_x[1] - spread, best_x[1] + spread),
                (best_x[2] - spread * 0.5, best_x[2] + spread * 0.5),
            ]
            res = differential_evolution(
                _objective, bounds_tight, seed=123, maxiter=300, popsize=30,
                tol=1e-14, polish=False, init='sobol'
            )
            if res.fun < best_v:
                best_v = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Stage 4: Basin hopping from best and seed
    # ------------------------------------------------------------------
    for x0 in [best_x, _SEED]:
        if _time_left() < 20:
            break
        bx, bv = _basin_hopping_run(x0, n_iter=30)
        if bv is not None and bv < best_v:
            best_v = bv
            best_x = bx

    # ------------------------------------------------------------------
    # Stage 5: Final high-precision polish
    # ------------------------------------------------------------------
    if _time_left() > 10:
        px, pv = _polish(best_x)
        if pv < best_v:
            best_v = pv
            best_x = px

    return np.array([best_x[0], best_x[1], best_x[2]])
