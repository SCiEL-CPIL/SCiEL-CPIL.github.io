"""Combined: analytic Hermite-Gaussian uncertainty bound + multi-start Nelder-Mead + high-precision polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200  # 10s safety margin before the 210s kill


def _hermite(x):
    """Physicists' Hermite polynomials H0, H4, H8, H12 evaluated at x."""
    x = np.asarray(x, dtype=np.float64)
    x2 = x * x
    H0 = np.ones_like(x)
    H4 = x2 * x2 - 6.0 * x2 + 3.0
    H8 = x2**4 - 28.0 * x2**3 + 210.0 * x2**2 - 420.0 * x2 + 105.0
    H12 = x2**6 - 66.0 * x2**5 + 1155.0 * x2**4 - 6930.0 * x2**3 + 13860.0 * x2**2 - 7560.0 * x2 + 945.0
    return H0, H4, H8, H12


def _moments(c):
    """
    Exact even moments of |g|^2 and |G|^2 for g(x)=P(x)e^{-x^2/2},
    G(xi)=sqrt(2 pi) P(-i xi) e^{-xi^2/2}.
    Returns (m0, m2, m4) for each distribution.
    """
    a0, a4, a8, a12 = c
    # g: real even moments
    m0 = (a0 * a0 + 3.0 * a4 * a4 + 105.0 * a8 * a8 + 945.0 * a12 * a12) * np.pi**0.5
    m2 = (a0 * (3.0 * a0 + 12.0 * a4)
          + a4 * (12.0 * a0 + 105.0 * a4 + 840.0 * a8)
          + a8 * (840.0 * a4 + 2940.0 * a8)
          + 11340.0 * a8 * a12) * np.pi**0.5
    m4 = (a0 * (15.0 * a0 + 60.0 * a4)
          + a4 * (60.0 * a0 + 105.0 * a4 + 840.0 * a8)
          + a8 * (840.0 * a4 + 2940.0 * a8 + 45360.0 * a12)
          + 11340.0 * a8 * a12) * np.pi**0.5
    # G: complex even moments (conjugate symmetry)
    g0 = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    g2 = a0 - 3.0 * a4 + 105.0 * a8 - 945.0 * a12
    g4 = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    n0 = 2.0 * np.pi * (g0 * g0 + 9.0 * g2 * g2 + 11025.0 * g4 * g4)
    n2 = 2.0 * np.pi * (3.0 * g0 * g0 + 21.0 * g2 * g2 + 2940.0 * g4 * g4)
    n4 = 2.0 * np.pi * (15.0 * g0 * g0 + 105.0 * g2 * g2 + 2940.0 * g4 * g4)
    return (m0, m2, m4), (n0, n2, n4)


def _bound(a0, a4, a8):
    """Uncertainty product A(f)*A(F) for the Hermite-Gaussian combination."""
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    (m0, m2, m4), (n0, n2, n4) = _moments((a0, a4, a8, a12))
    if m0 <= 0.0 or n0 <= 0.0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18
    vf = m2 / m0
    vF = n2 / n0
    if vf <= 0.0 or vF <= 0.0:
        return 1e18
    return np.sqrt(vf * vF)


def _objective(params):
    v = _bound(params[0], params[1], params[2])
    return v if np.isfinite(v) else 1e18


def construct():
    seed = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240.0])
    best_val = _bound(*seed)
    best = seed.copy()

    rng = np.random.default_rng(42)
    starts = [seed.copy()]
    for i in range(12):
        scale = 0.15 / (i + 1)
        starts.append(best + rng.normal(0.0, scale, 3))

    for s in starts:
        if time.time() > _DEADLINE - 15:
            break
        try:
            res = minimize(_objective, s, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best = res.x.copy()
        except Exception:
            pass

    # High-precision polish pass with tighter tolerances
    if time.time() < _DEADLINE - 8:
        try:
            res = minimize(_objective, best, method='Nelder-Mead',
                           options={'maxiter': 20000, 'xatol': 1e-15, 'fatol': 1e-17})
            if res.fun < best_val:
                best_val = res.fun
                best = res.x.copy()
        except Exception:
            pass

    return np.array(best)
# EVOLVE-BLOCK-END
