"""Optimized program for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hermite(n, x):
    """Compute the n-th probabilists' Hermite polynomial He_n(x)."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x
    H0 = np.ones_like(x)
    H1 = x
    for k in range(2, n + 1):
        H2 = x * H1 - (k - 1) * H0
        H0, H1 = H1, H2
    return H1

def _compute_objective(coeffs):
    """Compute the upper bound on A(f)*A(Fourier(f)) for given (a0, a4, a8)."""
    a0, a4, a8 = coeffs
    # a12 from g(0) = 0: a0 + 3*a4 + 105*a8 + 10395*a12 = 0
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 10395.0
    
    # Use a fine grid for numerical integration
    x = np.linspace(-15, 15, 100001)
    dx = x[1] - x[0]
    
    # Compute Hermite polynomials up to degree 12
    H0 = np.ones_like(x)
    H1 = x
    H2 = x * H1 - H0
    H3 = x * H2 - 2 * H1
    H4 = x * H3 - 3 * H2
    H5 = x * H4 - 4 * H3
    H6 = x * H5 - 5 * H4
    H7 = x * H6 - 6 * H5
    H8 = x * H7 - 7 * H6
    H9 = x * H8 - 8 * H7
    H10 = x * H9 - 9 * H8
    H11 = x * H10 - 10 * H9
    H12 = x * H11 - 11 * H10
    
    # g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    poly = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
    g = poly * np.exp(-0.5 * x**2)
    
    # A(g) = integral of |g(x)| dx
    Ag = np.sum(np.abs(g)) * dx
    
    # Fourier transform using the property that F[He_n(x)*exp(-x^2/2)] = (-i)^n * He_n(xi) * exp(-xi^2/2)/sqrt(2*pi)
    # For even n, (-i)^n is real: (-i)^0=1, (-i)^4=1, (-i)^8=1, (-i)^12=1
    # So the Fourier transform is purely real for our even-degree polynomials
    xi = x  # symmetric grid
    H0_f = np.ones_like(xi)
    H1_f = xi
    H2_f = xi * H1_f - H0_f
    H3_f = xi * H2_f - 2 * H1_f
    H4_f = xi * H3_f - 3 * H2_f
    H5_f = xi * H4_f - 4 * H3_f
    H6_f = xi * H5_f - 5 * H4_f
    H7_f = xi * H6_f - 6 * H5_f
    H8_f = xi * H7_f - 7 * H6_f
    H9_f = xi * H8_f - 8 * H7_f
    H10_f = xi * H9_f - 9 * H8_f
    H11_f = xi * H10_f - 10 * H9_f
    H12_f = xi * H11_f - 11 * H10_f
    
    # F(g)(xi) = (1/sqrt(2*pi)) * (a0*H0_f + a4*H4_f + a8*H8_f + a12*H12_f) * exp(-xi^2/2)
    # All coefficients are real and (-i)^n = 1 for even n
    poly_f = a0 * H0_f + a4 * H4_f + a8 * H8_f + a12 * H12_f
    g_f = (1.0 / np.sqrt(2 * np.pi)) * poly_f * np.exp(-0.5 * xi**2)
    
    # A(F(g)) = integral of |F(g)(xi)| dxi
    A_Fg = np.sum(np.abs(g_f)) * dx
    
    return Ag * A_Fg

def construct():
    start_time = time.time()
    deadline = start_time + 195.0  # safety margin before 210s limit
    
    # Initial guess from prior art
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Try multiple optimization strategies
    best_val = _compute_objective(x0)
    best_x = x0.copy()
    
    # Strategy 1: Standard L-BFGS-B optimization
    try:
        result = minimize(
            _compute_objective, x0, method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_x
    
    # Strategy 2: Try different initial points
    initial_points = [
        np.array([-1.0, 0.04, 0.0003]),
        np.array([-1.2, 0.05, 0.0002]),
        np.array([-0.8, 0.03, 0.0004]),
        np.array([-1.5, 0.06, 0.0001]),
        np.array([-1.0, 0.05, 0.0002]),
        np.array([-0.9, 0.035, 0.00035]),
        np.array([-1.1, 0.045, 0.00025]),
        np.array([-1.3, 0.055, 0.00015]),
    ]
    
    for ip in initial_points:
        if time.time() > deadline:
            break
        try:
            result = minimize(
                _compute_objective, ip, method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    if time.time() > deadline:
        return best_x
    
    # Strategy 3: Nelder-Mead for local refinement (handles non-smoothness)
    try:
        result = minimize(
            _compute_objective, best_x, method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_x
    
    # Strategy 4: Final polish with BFGS
    try:
        result = minimize(
            _compute_objective, best_x, method='BFGS',
            options={'maxiter': 200, 'gtol': 1e-12}
        )
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    return best_x
# EVOLVE-BLOCK-END
