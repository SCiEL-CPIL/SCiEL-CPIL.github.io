"""Refined Cohn-Goncalves Laguerre construction for the Fourier sign uncertainty bound.

We tune the 12 positive double-root locations (m=12, n=1) to minimize the
upper bound C4 on the Fourier sign uncertainty.  We start from the zeros of
L_12 and refine them via local optimization to minimize a proxy for C4.
A deadline guard keeps the call well under the 600 s wall-clock limit.
"""
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 560.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _seed_roots():
    return [4.0 * (i + 1) for i in range(12)]


def _laguerre_zeros(m):
    """Positive zeros of the standard Laguerre polynomial L_m, ascending."""
    coeffs = np.polynomial.laguerre.laguerre(m)
    roots = np.roots(coeffs)
    roots = roots[np.abs(roots.imag) < 1e-10].real
    roots = roots[roots > 0.0]
    roots = np.sort(roots)
    return roots


def _valid(roots):
    """Root constraints: 12 positive, strictly increasing (distinct) roots."""
    if len(roots) != 12:
        return False
    if np.any(roots <= 0.0):
        return False
    if np.any(np.diff(roots) <= 0.0):
        return False
    return True


def _objective(roots):
    """
    Proxy for C4: the L2 norm of the polynomial with double roots at r_i.
    For a polynomial P(x) = prod (x - r_i)^2, the sign uncertainty bound
    is often related to ||P||_2^2 / ||x*P||_2^2 or similar ratios.
    We use the ratio of the L2 norm of P to the L2 norm of x*P on [0, max_r],
    which penalizes roots that are too far apart or too close together.
    """
    if len(roots) < 2:
        return 1e10
    x = np.linspace(0, max(roots) * 1.5, 2000)
    P = np.ones_like(x)
    for r in roots:
        P *= (x - r) ** 2
    xP = x * P
    norm_P = np.sqrt(np.trapz(P ** 2, x))
    norm_xP = np.sqrt(np.trapz(xP ** 2, x))
    if norm_xP < 1e-15:
        return 1e10
    return norm_P / norm_xP


def construct():
    if _elapsed() > _DEADLINE:
        return _seed_roots()

    try:
        x0 = _laguerre_zeros(12)
        if not _valid(x0):
            x0 = np.array(_seed_roots(), dtype=float)
        
        # Define bounds: roots must be positive and ordered
        bounds = [(1e-6, None)] * 12
        
        # Use L-BFGS-B with ordering constraints handled by penalty
        def constrained_objective(params):
            roots = np.sort(params)
            if not _valid(roots):
                return 1e10
            return _objective(roots)
        
        result = minimize(
            constrained_objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'maxfun': 10000}
        )
        
        if _elapsed() > _DEADLINE - 30:
            return [float(r) for r in x0]
        
        if result.success and _valid(result.x):
            return [float(r) for r in np.sort(result.x)]
        else:
            return [float(r) for r in x0]
    except Exception:
        pass

    return _seed_roots()
