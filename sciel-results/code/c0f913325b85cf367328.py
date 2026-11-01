"""Rewritten program: minimize the uncertainty upper bound over Hermite coefficients."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 210.0
_MARGIN = 15.0

# Probabilists' Hermite polynomials evaluated as polynomials.
def _hermite(k, x):
    # Use recurrence for numerical stability.
    n = len(x)
    H0 = np.ones(n)
    if k == 0:
        return H0
    H1 = x.copy()
    if k == 1:
        return H1
    for m in range(2, k + 1):
        H2 = x * H1 - (m - 1) * H0
        H0, H1 = H1, H2
    return H1

def _coeffs_to_g(c0, c4, c8):
    # Build polynomial p(x) = c0 H0 + c4 H4 + c8 H8 + c12 H12, with p(0)=0.
    # p(0) = c0*1 + c4*3 + c8*105 + c12*945 = 0  => c12 = -(c0 + 3 c4 + 105 c8)/945
    c12 = -(c0 + 3.0 * c4 + 105.0 * c8) / 945.0
    return c0, c4, c8, c12

def _poly_and_deriv(c0, c4, c8, c12, x):
    # p(x) and p'(x) as polynomials in x.
    # H0 = 1
    # H4 = x^4 - 6 x^2 + 3
    # H8 = x^8 - 28 x^6 + 210 x^4 - 420 x^2 + 105
    # H12 = x^12 - 54 x^10 + 1080 x^8 - 7560 x^6 + 15120 x^4 - 7560 x^2 + 945
    # p(x) = a12 x^12 + a10 x^10 + a8 x^8 + a6 x^6 + a4 x^4 + a2 x^2 + a0
    a12 = c12
    a10 = -54.0 * c12
    a8  = c8 - 28.0 * c12 + 1080.0 * c12
    a6  = -7560.0 * c12
    a4  = c4 + 210.0 * c8 - 420.0 * c12 + 15120.0 * c12
    a2  = -6.0 * c4 - 420.0 * c8 + 105.0 * c12 - 7560.0 * c12
    a0  = c0 + 3.0 * c4 + 105.0 * c8 + 945.0 * c12  # = 0 by construction
    # p'(x) = 12 a12 x^11 + 10 a10 x^9 + 8 a8 x^7 + 6 a6 x^5 + 4 a4 x^3 + 2 a2 x
    p = (((((a12*x + a10)*x + a8)*x + a6)*x + a4)*x + a2)*x + a0
    dp = (((((12*a12*x + 10*a10)*x + 8*a8)*x + 6*a6)*x + 4*a4)*x + 2*a2)*x
    return p, dp

def _bound(c0, c4, c8, grid):
    c12 = _coeffs_to_g(c0, c4, c8)[3]
    x = grid
    p, dp = _poly_and_deriv(c0, c4, c8, c12, x)
    w = np.exp(-x*x/2.0)
    g = p * w
    gp = (dp - x * p) * w   # derivative of p(x) e^{-x^2/2}
    I0 = np.trapz(g*g, x)
    if I0 <= 1e-30:
        return 1e300
    I1 = np.trapz(gp*gp, x)
    I2 = np.trapz(x*x*g*g, x)
    # Heisenberg-type upper bound on the uncertainty product
    return 0.25 * (I1 / I0) * (I2 / I0)

def _make_grid():
    # Symmetric grid covering the Gaussian support.
    x = np.linspace(-12.0, 12.0, 2401)
    return x

def construct():
    t0 = time.time()
    grid = _make_grid()

    # Seed from the prior state of the art.
    best_c = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = _bound(*best_c, grid)

    def obj(c):
        return _bound(c[0], c[1], c[2], grid)

    # Multistart local optimization with a deadline guard.
    starts = [best_c.copy()]
    rng = np.random.default_rng(0)
    for _ in range(20):
        starts.append(rng.normal(0.0, 1.0, size=3))
    # A few structured starts around the seed.
    for scale in (0.5, 2.0, -1.0):
        starts.append(best_c * scale)

    for s in starts:
        if time.time() - t0 > _DEADLINE - _MARGIN:
            break
        try:
            res = minimize(obj, s, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.success or (res.fun < best_val):
                if res.fun < best_val:
                    best_val = res.fun
                    best_c = res.x.copy()
        except Exception:
            pass

    # High-precision polish pass if time remains.
    if time.time() - t0 < _DEADLINE - _MARGIN:
        try:
            res = minimize(obj, best_c, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-15})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass

    return np.array([best_c[0], best_c[1], best_c[2]])
# EVOLVE-BLOCK-END
