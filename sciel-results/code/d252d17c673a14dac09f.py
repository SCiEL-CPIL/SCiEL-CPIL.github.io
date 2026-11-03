"""Rewritten program: use a global search (differential evolution) with a
tighter, precomputed evaluation to minimise the uncertainty-product upper bound.

The test function is
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
with c12 fixed so that g(0) = 0, i.e.
    c0 - 3*c4 + 105*c8 - 945*c12 = 0   =>   c12 = (c0 - 3*c4 + 105*c8)/945.

The quantity to minimise is the upper bound on A(f)*A(Fourier(f)).
For this class of functions the product is (M2_x)^2 where
    M2_x = <x^2> = <x^2 P(x)^2> / <P(x)^2>
and the integrals are w.r.t. the standard Gaussian weight (Gauss-Hermite).

We use a larger Gauss-Hermite quadrature for accuracy, precompute the
Hermite values at the nodes, and run a global optimiser (differential
evolution) followed by a local polish (Nelder-Mead) within a strict
time budget.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import eval_hermite
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize, differential_evolution

# ---------------------------------------------------------------------------
# Precomputation: Gauss-Hermite quadrature and Hermite polynomials at nodes.
# P(x) has degree 12, so P(x)^2 has degree 24, and x^2*P(x)^2 has degree 26.
# N=25 integrates polynomials up to degree 2*25-1 = 49 exactly.
# ---------------------------------------------------------------------------
N_QUAD = 25
_x_nodes, _w_weights = hermgauss(N_QUAD)
_x_sq = _x_nodes ** 2

_H0 = np.ones(N_QUAD)
_H4 = eval_hermite(4, _x_nodes)
_H8 = eval_hermite(8, _x_nodes)
_H12 = eval_hermite(12, _x_nodes)

def _get_c12(c0, c4, c8):
    """c12 such that P(0) = 0."""
    return (c0 - 3.0 * c4 + 105.0 * c8) / 945.0

def _compute_bound(c0, c4, c8):
    """Return (M2_x)^2, the upper bound on A(f)*A(Fourier(f))."""
    c12 = _get_c12(c0, c4, c8)
    p_vals = c0 * _H0 + c4 * _H4 + c8 * _H8 + c12 * _H12
    p_sq = p_vals * p_vals
    den = np.dot(_w_weights, p_sq)
    if den < 1e-15:
        return 1e10
    num = np.dot(_w_weights, _x_sq * p_sq)
    m2 = num / den
    return m2 * m2

def construct():
    t_start = time.time()
    deadline = t_start + 190.0   # safety margin before 210 s kill

    # Baseline: the seed coefficients.
    c_base = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    def objective(c):
        return _compute_bound(c[0], c[1], c[2])

    best_c = c_base.copy()
    best_val = objective(best_c)

    # ------------------------------------------------------------------
    # Phase 1: Global search with differential evolution.
    # Search in a box around the baseline; the optimal coefficients are
    # expected to be O(1) or smaller.
    # ------------------------------------------------------------------
    bounds = [(-5.0, 5.0), (-2.0, 2.0), (-0.5, 0.5)]
    try:
        res_de = differential_evolution(
            objective,
            bounds=bounds,
            seed=42,
            maxiter=200,
            tol=1e-10,
            polish=False,
            init='sobol',
            popsize=15,
        )
        if res_de.fun < best_val:
            best_val = res_de.fun
            best_c = res_de.x
    except Exception:
        pass

    if time.time() > deadline - 10.0:
        return np.array(best_c)

    # ------------------------------------------------------------------
    # Phase 2: Local polish with Nelder-Mead from the best global point.
    # ------------------------------------------------------------------
    try:
        res_nm = minimize(
            objective,
            best_c,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14,
                     'adaptive': True},
        )
        if res_nm.fun < best_val:
            best_val = res_nm.fun
            best_c = res_nm.x
    except Exception:
        pass

    if time.time() > deadline - 5.0:
        return np.array(best_c)

    # ------------------------------------------------------------------
    # Phase 3: A second DE run from a slightly different initialisation
    # to guard against local minima, if time remains.
    # ------------------------------------------------------------------
    try:
        res_de2 = differential_evolution(
            objective,
            bounds=bounds,
            seed=12345,
            maxiter=100,
            tol=1e-10,
            polish=False,
            init='latinhypercube',
            popsize=12,
        )
        if res_de2.fun < best_val:
            best_val = res_de2.fun
            best_c = res_de2.x
        # Quick local polish from this new point too.
        res_nm2 = minimize(
            objective,
            best_c,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14,
                     'adaptive': True},
        )
        if res_nm2.fun < best_val:
            best_val = res_nm2.fun
            best_c = res_nm2.x
    except Exception:
        pass

    return np.array(best_c)

# EVOLVE-BLOCK-END
