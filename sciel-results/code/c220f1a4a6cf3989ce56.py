"""Optimized search for Hermite-Gaussian uncertainty inequality coefficients."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_hermite
import time

_START_TIME = time.time()
_TIME_LIMIT = 200.0  # Safety margin before 210s kill

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_A(a0, a4, a8):
    """Compute A(g) = integral of |g(x)| dx where g = (a0*H0 + a4*H4 + a8*H8 + a12*H12)*exp(-x^2/2)"""
    # Hermite polynomials at 0: H0(0)=1, H4(0)=-3, H8(0)=105, H12(0)=-10395
    a12 = (a0 - 3*a4 + 105*a8) / 10395.0
    
    # Use Gauss-Hermite quadrature for the Gaussian-weighted integral
    # A(g) = integral |g(x)| dx = integral |sum an*Hn(x)| * exp(-x^2/2) dx
    # Let u = x/sqrt(2), then x = sqrt(2)*u, dx = sqrt(2)du
    # exp(-x^2/2) = exp(-u^2)
    # A(g) = sqrt(2) * integral |sum an*Hn(sqrt(2)*u)| * exp(-u^2) du
    
    # Gauss-Hermite quadrature: integral f(u)*exp(-u^2) du ≈ sum w_i f(u_i)
    # We need enough points for the polynomial to be well-resolved
    n_points = 200
    u, w = np.polynomial.hermite.hermgauss(n_points)
    x = np.sqrt(2) * u
    
    H0 = np.ones_like(x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    
    g_vals = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
    
    # A(g) = sqrt(2) * sum w_i * |g_vals_i|
    A = np.sqrt(2) * np.sum(w * np.abs(g_vals))
    return A

def _objective(coeffs):
    """Objective: A(g)^2 (since A(g)=A(hat g))"""
    a0, a4, a8 = coeffs
    A = _compute_A(a0, a4, a8)
    return A * A

def construct():
    # Start from the known best solution
    best_coeffs = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = _objective(best_coeffs)
    
    # Try local optimization from the starting point
    if _time_left() > 10:
        try:
            result = minimize(_objective, best_coeffs, method='Nelder-Mead',
                            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
            if result.fun < best_val:
                best_coeffs = result.x
                best_val = result.fun
        except Exception:
            pass
    
    # Try multiple random starts with local optimization
    np.random.seed(42)
    for trial in range(50):
        if _time_left() < 5:
            break
        # Random initialization in a reasonable range
        start = np.random.uniform(-5, 5, size=3)
        try:
            result = minimize(_objective, start, method='Nelder-Mead',
                            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    # Try differential evolution if time permits
    if _time_left() > 30:
        try:
            bounds = [(-10, 10), (-10, 10), (-10, 10)]
            de_result = differential_evolution(_objective, bounds, maxiter=100,
                                               popsize=15, tol=1e-10, seed=42,
                                               polish=True, init='latinhypercube')
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_coeffs = de_result.x
        except Exception:
            pass
    
    # Final polish
    if _time_left() > 5:
        try:
            result = minimize(_objective, best_coeffs, method='Nelder-Mead',
                            options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16})
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    return np.array(best_coeffs)
