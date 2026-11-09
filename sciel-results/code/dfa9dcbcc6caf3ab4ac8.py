"""
Refined Cohn-Goncalves Laguerre-polynomial construction for the Fourier sign
uncertainty upper bound (m=12, n=1).

We search for 12 positive double-root locations r_1 < ... < r_12 that
minimise the C4 upper bound.  The bound is evaluated using Gauss-Laguerre
quadrature on the associated Laguerre combination.

Strategy:
  1. Compute an analytic (or numerically robust) surrogate objective that
     captures the dependence of C4 on the root configuration.
  2. Use a multi-start local optimiser (scipy.optimize.minimize) seeded from
     several geometrically motivated initial guesses (Laguerre nodes,
     arithmetic, logarithmic, etc.).
  3. Return the best root configuration found within a strict time budget.
"""

import time
import math
import numpy as np
from scipy.special import roots_laguerre, eval_genlaguerre
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 580.0  # leave margin below the 600 s evaluator limit


def _elapsed():
    return time.time() - _START_TIME


def _deadline_reached():
    return _elapsed() > _TIME_LIMIT


def _laguerre_combination(r, m=12, n=1, x=None):
    """
    Evaluate the Laguerre combination at points x.

    The combination is  P(x) = prod_i (x - r_i)^2  *  L_m^{(n)}(x)
    but in the Cohn-Goncalves context the "Laguerre combination" whose
    sign changes matter is typically  Q(x) = L_m^{(n)}(x) + c * P(x) * L_{m-1}^{(n+2)}(x)
    or similar.  For the purpose of the upper bound on C4, the critical
    quantity is the supremum of  |P(x) * L_m^{(n)}(x)| * e^{-x} / x^{n/2}
    or a related weighted norm.

    A more direct formulation used in the literature:
      C4 ~ sup_x  |  prod_i (x - r_i)^2  *  e^{-x/2}  *  L_m^{(n)}(x)  |
    We optimise  log(sup)  via quadrature-based estimation.
    """
    if x is None:
        return None
    # P(x) = prod (x - r_i)^2
    P = np.ones_like(x, dtype=np.float64)
    for ri in r:
        P *= (x - ri) ** 2
    # L_m^{(n)}(x)
    Lm = eval_genlaguerre(m, n, x)
    # weight
    w = np.exp(-0.5 * x) * x ** (n / 2.0)
    return P * Lm * w


def _compute_bound(r, m=12, n=1, n_quad=256):
    """
    Estimate the C4 upper bound via Gauss-Laguerre quadrature.

    The bound involves the L-infinity (or related) norm of
        f(x) = prod_i (x - r_i)^2 * L_m^{(n)}(x) * e^{-x/2} * x^{n/2}
    over x in (0, infinity).  We approximate the supremum by evaluating
    on a fine grid including the Laguerre nodes and their intervals.
    """
    if _deadline_reached():
        return float('inf')

    # Use Gauss-Laguerre nodes and a supplementary grid
    x_gl, w_gl = roots_laguerre(n_quad)

    # Also add a log-spaced grid to catch peaks between nodes
    x_grid = np.geomspace(1e-4, max(2.0 * r[-1] + 10, 100.0), 512)
    x_all = np.concatenate([x_gl, x_grid])

    f = _laguerre_combination(r, m=m, n=n, x=x_all)
    # The bound is related to the max of |f|
    bound = np.max(np.abs(f))
    return bound


def _objective(r, m=12, n=1):
    """Objective to minimise: the C4 upper bound."""
    if _deadline_reached():
        return 1e12
    # Ensure positivity and ordering via transformation is handled externally;
    # here r is already positive and sorted.
    b = _compute_bound(r, m=m, n=n)
    if not np.isfinite(b):
        return 1e12
    return b


def _unconstrained_objective(y, m=12, n=1):
    """
    Objective in unconstrained space.  y -> r via r_i = exp(y_i),
    which guarantees positivity.  We do NOT enforce ordering here;
    the objective is symmetric in the roots (product of squares), so
    the optimiser can permute them freely.
    """
    if _deadline_reached():
        return 1e12
    r = np.exp(y)
    return _objective(r, m=m, n=n)


def _initial_guesses(m=12):
    """Generate several diverse initial guesses in log-space."""
    guesses = []

    # 1. Zeros of L_m(x)  (Gauss-Laguerre nodes)
    x_lag, _ = roots_laguerre(m)
    guesses.append(np.log(x_lag + 1e-12))

    # 2. Zeros of L_m^{(1)}(x)  (generalised, alpha=1)
    from scipy.special import roots_genlaguerre
    x_lag1, _ = roots_genlaguerre(m, 1)
    guesses.append(np.log(x_lag1 + 1e-12))

    # 3. Arithmetic spacing similar to seed: 4,8,...,48
    r_arith = 4.0 * np.arange(1, m + 1, dtype=np.float64)
    guesses.append(np.log(r_arith))

    # 4. Logarithmic spacing from 1 to 50
    r_log = np.geomspace(1.0, 50.0, m)
    guesses.append(np.log(r_log))

    # 5. Scaled Laguerre nodes: match max to 48
    x_lag, _ = roots_laguerre(m)
    scale = 48.0 / x_lag.max()
    guesses.append(np.log(x_lag * scale + 1e-12))

    # 6. Scaled L_m^{(1)} nodes
    x_lag1, _ = roots_genlaguerre(m, 1)
    scale1 = 48.0 / x_lag1.max()
    guesses.append(np.log(x_lag1 * scale1 + 1e-12))

    return guesses


def _construct_core(m=12, n=1, time_budget=500.0):
    """
    Multi-start local optimisation over the 12 log-roots.
    Returns the best root configuration found.
    """
    best_r = None
    best_val = float('inf')

    guesses = _initial_guesses(m)

    for gi, y0 in enumerate(guesses):
        if _deadline_reached():
            break
        t0 = time.time()

        try:
            res = minimize(
                _unconstrained_objective,
                y0,
                args=(m, n),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10,
                         'maxls': 100},
            )
        except Exception:
            continue

        if _deadline_reached():
            break

        if res.success or res.fun < best_val:
            y_opt = res.x
            r_opt = np.sort(np.exp(y_opt))
            val = _objective(r_opt, m=m, n=n)
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_r = r_opt

        # If we have time, try a second, more aggressive pass from the best
        if gi == 0 and not _deadline_reached():
            try:
                res2 = minimize(
                    _unconstrained_objective,
                    y0,
                    args=(m, n),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10,
                             'adaptive': True},
                )
                if res2.fun < best_val:
                    y_opt2 = res2.x
                    r_opt2 = np.sort(np.exp(y_opt2))
                    val2 = _objective(r_opt2, m=m, n=n)
                    if np.isfinite(val2) and val2 < best_val:
                        best_val = val2
                        best_r = r_opt2
            except Exception:
                pass

    if best_r is None:
        # Fallback: Laguerre nodes
        x_lag, _ = roots_laguerre(m)
        best_r = np.sort(x_lag)

    return best_r


def construct():
    """
    Return a list of 12 positive double-root locations for the Laguerre
    combination (m=12, n=1) that minimises the C4 upper bound.
    """
    m, n = 12, 1
    r = _construct_core(m=m, n=n)
    # Ensure it is a plain Python list of floats
    return [float(x) for x in r]
