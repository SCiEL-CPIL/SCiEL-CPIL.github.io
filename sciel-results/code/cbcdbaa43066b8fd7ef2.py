"""Optimized program for the uncertainty-inequality (C4) tuning task."""
import time
import numpy as np
from scipy.optimize import minimize

# Hermite polynomials (physicists' convention) in powers of x.
_HERMITE = {
    0: np.array([1.0]),
    4: np.array([12.0, 0.0, -48.0, 0.0, 24.0]),
    8: np.array([1344.0, 0.0, -6720.0, 0.0, 13440.0, 0.0, -13440.0, 0.0, 3024.0]),
    12: np.array([40840.0, 0.0, -306300.0, 0.0, 692640.0, 0.0, -774144.0, 0.0,
                  479040.0, 0.0, -145152.0, 0.0, 10321.0]),
}


def _eval_poly(coeffs, x):
    """Evaluate a polynomial given in ascending power order."""
    return np.polynomial.polynomial.polyval(x, coeffs)


def _poly_product(a, b):
    """Product of two polynomials (ascending power order)."""
    return np.polynomial.polynomial.polymul(a, b)


def _poly_integrate(a, b):
    """Integral of product of two polynomials over the real line."""
    prod = _poly_product(a, b)
    # integral x^n = 0 for odd n, and for even n=2k: 2^k * k! * sqrt(pi)
    total = 0.0
    for n in range(0, len(prod), 2):
        k = n // 2
        total += prod[n] * (2.0 ** k) * float(np.math.factorial(k))
    return total * np.sqrt(np.pi)


def _norm_sq(poly):
    return _poly_integrate(poly, poly)


def get_c12(c0, c4, c8):
    """Coefficient of H12 chosen so that g(0) = 0."""
    # g(0) = c0*1 + c4*12 + c8*1344 + c12*40840 = 0
    return -(c0 + 12.0 * c4 + 1344.0 * c8) / 40840.0


def objective(x):
    """Upper bound on C4 (A(f)*A(F)) for the given (c0, c4, c8)."""
    c0, c4, c8 = x
    c12 = get_c12(c0, c4, c8)

    # Build the full polynomial g(x) = sum c_k H_k(x).
    g = (c0 * _HERMITE[0] + c4 * _HERMITE[4] + c8 * _HERMITE[8] + c12 * _HERMITE[12])
    g = np.pad(g, (0, max(0, 13 - len(g))))

    # g^2(x) = sum_k 2^k (g_k)^2 H_k(x)
    g2_coeffs = np.zeros(25)
    for k in range(len(g)):
        if k % 2 == 0:
            g2_coeffs[k] += (2.0 ** k) * (g[k] ** 2)

    # Fourier transform of H_k(x) e^{-x^2/2} = (-1)^k H_k(x) e^{-x^2/2}
    ghat = g.copy()
    for k in range(1, len(ghat), 2):
        ghat[k] = -ghat[k]

    # ghat^2(x) = sum_k 2^k (ghat_k)^2 H_k(x)
    ghat2_coeffs = np.zeros(25)
    for k in range(len(ghat)):
        if k % 2 == 0:
            ghat2_coeffs[k] += (2.0 ** k) * (ghat[k] ** 2)

    # Integrals
    I1 = _poly_integrate(g2_coeffs, g2_coeffs)          # ∫ g^4
    I2 = _poly_integrate(ghat2_coeffs, ghat2_coeffs)    # ∫ ĝ^4
    I3 = _poly_integrate(g2_coeffs, ghat2_coeffs)       # ∫ g^2 ĝ^2
    I4 = _poly_integrate(g2_coeffs, ghat2_coeffs)       # same as I3

    # A(f) = (∫g^4)^{1/4} (∫ĝ^4)^{1/4}; A(F) = (∫g^2 ĝ^2)^{1/2}
    A_f = (I1 * I2) ** 0.25
    A_F = np.sqrt(I3)
    return A_f * A_F


def construct():
    deadline = time.time() + 180.0
    best_x = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val = objective(best_x)

    def time_left():
        return deadline - time.time()

    # Start points: the prior-art solution and a spread of candidates.
    starts = [best_x.copy()]
    rng = np.random.default_rng(12345)
    for _ in range(24):
        starts.append(rng.normal(0.0, 1.0, size=3))
    # A few targeted directions around the prior art.
    for scale in (0.1, 0.5, 2.0):
        starts.append(best_x + scale * np.array([-1.0, 1.0, 1.0]))
        starts.append(best_x + scale * np.array([1.0, -1.0, -1.0]))

    for x0 in starts:
        if time_left() < 15.0:
            break
        try:
            res = minimize(
                objective,
                x0,
                method="L-BFGS-B",
                options={"maxiter": 400, "ftol": 1e-16, "gtol": 1e-12,
                         "maxfun": 20000},
            )
            if res.x is not None:
                val = objective(res.x)
                if np.isfinite(val) and val < best_val:
                    best_val = val
                    best_x = res.x.copy()
        except Exception:
            continue

    # High-precision polish pass on the best point found.
    if time_left() > 10.0:
        try:
            res = minimize(
                objective,
                best_x,
                method="L-BFGS-B",
                options={"maxiter": 2000, "ftol": 1e-20, "gtol": 1e-15,
                         "maxfun": 100000},
            )
            if res.x is not None:
                val = objective(res.x)
                if np.isfinite(val) and val < best_val:
                    best_val = val
                    best_x = res.x.copy()
        except Exception:
            pass

    if time_left() < 5.0:
        # Safety margin: return best found so far.
        return np.array(best_x)

    return np.array(best_x)
