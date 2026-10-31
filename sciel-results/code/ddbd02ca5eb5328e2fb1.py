"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import quad
from scipy.optimize import minimize

_DEADLINE = time.time() + 195  # 15s margin before 210s kill


def _hermite_coeffs():
    """Return coefficients for H0, H4, H8, H12 evaluated at x=0."""
    # H0(0) = 1
    # H4(x) = x^4 - 6x^2 + 3, so H4(0) = 3
    # H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105, so H8(0) = 105
    # H12(x) = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945, so H12(0) = 945
    return 1.0, 3.0, 105.0, 945.0


def _compute_delta_x_squared(a0, a4, a8):
    """
    Compute (Δx)² for g(x) = P(x)exp(-x²/2) where
    P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12 with g(0)=0.
    
    For even functions, the uncertainty product = (Δx)².
    """
    if time.time() > _DEADLINE:
        return None

    h0_0, h4_0, h8_0, h12_0 = _hermite_coeffs()
    # Constraint g(0) = 0: a0*h0_0 + a4*h4_0 + a8*h8_0 + a12*h12_0 = 0
    a12 = -(a0 * h0_0 + a4 * h4_0 + a8 * h8_0) / h12_0

    def g_sq(x):
        # |g(x)|² = P(x)² * exp(-x²)
        p = (a0 * 1.0 +
             a4 * (x**4 - 6*x**2 + 3) +
             a8 * (x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105) +
             a12 * (x**12 - 66*x**10 + 1155*x**8 - 6930*x**6 + 13860*x**4 - 7560*x**2 + 945))
        return p * p * np.exp(-x * x)

    def x2_g_sq(x):
        return x * x * g_sq(x)

    try:
        norm_sq, _ = quad(g_sq, -np.inf, np.inf, limit=300, epsabs=1e-12, epsrel=1e-10)
        if norm_sq <= 0 or not np.isfinite(norm_sq):
            return None
        x2_norm, _ = quad(x2_g_sq, -np.inf, np.inf, limit=300, epsabs=1e-12, epsrel=1e-10)
        if not np.isfinite(x2_norm):
            return None
        delta_x_sq = x2_norm / norm_sq
        return delta_x_sq
    except Exception:
        return None


def construct():
    """Find optimal coefficients to minimize the uncertainty product bound."""
    start_time = time.time()

    # Initial guess from the seed (Goncalves et al. 2017)
    best_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val = _compute_delta_x_squared(*best_coeffs)
    if best_val is None:
        best_val = np.inf

    # Helper to check time
    def time_ok(margin=20):
        return (time.time() - start_time) < (195 - margin)

    # Strategy 1: Multi-start Nelder-Mead
    rng = np.random.default_rng(12345)
    n_starts = 30
    for i in range(n_starts):
        if not time_ok(30):
            break
        # Random perturbation around current best
        scale = 0.05 * (1 + i * 0.01)
        coeffs = best_coeffs + rng.normal(0, scale, 3)
        val = _compute_delta_x_squared(*coeffs)
        if val is not None and val < best_val:
            best_val = val
            best_coeffs = coeffs.copy()

    # Strategy 2: Nelder-Mead local optimization from best point
    if time_ok(40):
        def objective(x):
            if time.time() > _DEADLINE:
                return np.inf
            v = _compute_delta_x_squared(x[0], x[1], x[2])
            if v is None:
                return np.inf
            return v

        try:
            result = minimize(objective, best_coeffs, method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12,
                                       'adaptive': True})
            if np.isfinite(result.fun) and result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass

    # Strategy 3: Powell method for refinement
    if time_ok(30):
        def objective(x):
            if time.time() > _DEADLINE:
                return np.inf
            v = _compute_delta_x_squared(x[0], x[1], x[2])
            if v is None:
                return np.inf
            return v

        try:
            result = minimize(objective, best_coeffs, method='Powell',
                              options={'maxiter': 3000, 'xtol': 1e-12, 'ftol': 1e-14})
            if np.isfinite(result.fun) and result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass

    # Strategy 4: Final high-precision polish with very tight tolerances
    if time_ok(15):
        def objective_tight(x):
            if time.time() > _DEADLINE:
                return np.inf
            v = _compute_delta_x_squared(x[0], x[1], x[2])
            if v is None:
                return np.inf
            return v

        try:
            result = minimize(objective_tight, best_coeffs, method='Nelder-Mead',
                              options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16,
                                       'adaptive': True})
            if np.isfinite(result.fun) and result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass

    return best_coeffs
# EVOLVE-BLOCK-END
