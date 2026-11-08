"""Optimized search for the Hermite-Gaussian uncertainty inequality bound.

Structural change: rather than returning the fixed prior coefficients, we
numerically minimize the true uncertainty product  (Delta x)(Delta k)  over
the three free coefficients (a0, a4, a8) on a precomputed spatial grid.  The
degree-12 weight is fixed so that g(0)=0.  We use a robust multi-start
L-BFGS-B (with a coarse grid fallback) and an explicit wall-clock deadline
guard so the function always returns the best result found before the limit.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.special import eval_hermite

_DEADLINE = 210.0          # evaluator hard kill
_SAFETY = 8.0              # stop this many seconds before the kill
_T0 = time.perf_counter()

# ---- Precomputed spatial grid (fixed for the whole process) ---------------
# Symmetric, moderately wide grid so the Gaussian tails are negligible.
_N = 12001
_XMIN = -15.0
_XMAX = 15.0
_X = np.linspace(_XMIN, _XMAX, _N)
_DX = _X[1] - _X[0]
# Normalized probabilistic Gaussian weight (integrates to 1)
_W = np.exp(-0.5 * _X * _X)
_W = _W / np.sum(_W) * _DX          # so that sum(w*phi) ~ E[phi]
# Hermite polynomials evaluated once on the grid (physicists' H_n)
_H0 = eval_hermite(0, _X)
_H4 = eval_hermite(4, _X)
_H8 = eval_hermite(8, _X)
_H12 = eval_hermite(12, _X)
# Fourier-side weight.  With our convention,
#   |F(f)(k)|^2  ~  (1/2pi) * exp(-k^2/2)   (up to an irrelevant global const)
_K = _X                                   # same grid (symmetric problem)
_KW = np.exp(-0.5 * _K * _K)              # unnormalized Fourier weight


def _coeff12(a0, a4, a8):
    """Fix a12 so that g(0) = 0.

    g(x) = a0 H0 + a4 H4 + a8 H8 + a12 H12,  and g(0)=0.
    H_n(0) is 0 for odd n; for even n, H_n(0) = (-1)^(n/2) * (n-1)!!.
    """
    h0 = 1.0
    h4 = 3.0
    h8 = 105.0
    h12 = 945.0
    return -(a0 * h0 + a4 * h4 + a8 * h8) / h12


def _objective(coeffs):
    """Uncertainty product (Delta x)(Delta k) for the given (a0,a4,a8).

    Lower is better.  Uses the precomputed grid; vectorized and fast.
    """
    a0, a4, a8 = coeffs
    a12 = _coeff12(a0, a4, a8)

    # spatial wavefunction
    g = a0 * _H0 + a4 * _H4 + a8 * _H8 + a12 * _H12
    g2 = g * g
    g2w = g2 * _W

    norm = np.sum(g2w)
    if norm <= 1e-12:
        return 1e9
    # <x^2>  (x=0 by symmetry, so variance = <x^2>)
    x2 = _X * _X
    ex2 = np.sum(x2 * g2w) / norm

    # Fourier side
    f2w = _KW * g2
    fk = np.sum(f2w)
    if fk <= 1e-12:
        return 1e9
    k2 = _K * _K
    ek2 = np.sum(k2 * f2w) / fk

    return math.sqrt(max(ex2, 0.0)) * math.sqrt(max(ek2, 0.0))


def _grid_search():
    """Coarse multi-resolution grid fallback (also supplies a good start)."""
    best = None
    # coarse box around the prior point and a wider exploration box
    boxes = [
        (-2.0, 2.0, -1.0, 1.0, -0.5, 0.5, 30),
        (-6.0, 6.0, -3.0, 3.0, -1.5, 1.5, 25),
        (-12.0, 12.0, -6.0, 6.0, -3.0, 3.0, 22),
    ]
    for (x0, x1, y0, y1, z0, z1, n) in boxes:
        if time.perf_counter() - _T0 > _DEADLINE - _SAFETY:
            break
        gx = np.linspace(x0, x1, n)
        gy = np.linspace(y0, y1, n)
        gz = np.linspace(z0, z1, n)
        for a0 in gx:
            for a4 in gy:
                for a8 in gz:
                    v = _objective((a0, a4, a8))
                    if best is None or v < best[0]:
                        best = (v, (a0, a4, a8))
    return best


def _polish(start):
    """Local L-BFGS-B refinement of a starting point."""
    from scipy.optimize import minimize
    try:
        res = minimize(
            _objective,
            np.asarray(start, dtype=float),
            method="L-BFGS-B",
            options={"maxiter": 400, "maxfun": 8000, "ftol": 1e-12},
        )
        if res.success or res.fun < _objective(np.asarray(start, float)):
            return (res.fun, res.x)
    except Exception:
        pass
    v = _objective(np.asarray(start, float))
    return (v, np.asarray(start, float))


def construct():
    """Return the three free coefficients (a0, a4, a8) that minimize the
    uncertainty-product upper bound.  The degree-12 weight is fixed so g(0)=0.
    """
    deadline = _DEADLINE - _SAFETY

    # 1) coarse grid to get a robust starting point (and a safe fallback)
    best = _grid_search()
    if best is None:
        # ultimate fallback: prior state of the art
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val, best_pt = best

    # 2) polish the best grid point
    v, x = _polish(best_pt)
    if v < best_val:
        best_val, best_pt = v, x

    # 3) a couple of additional local refinements from nearby perturbations
    for i in range(6):
        if time.perf_counter() - _T0 > deadline:
            break
        scale = 0.3 * (1.0 + 0.0)
        rng = np.random.RandomState(1000 + i)
        cand = best_pt + rng.randn(3) * scale
        v2, x2 = _polish(cand)
        if v2 < best_val:
            best_val, best_pt = v2, x2

    return np.asarray(best_pt, dtype=float)
# EVOLVE-BLOCK-END
