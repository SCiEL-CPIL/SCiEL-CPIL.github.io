"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import quad
from scipy.optimize import minimize

DEADLINE = time.time() + 195  # 195s deadline, 15s margin before 210s kill

# Physicist's Hermite polynomials H_n(x)
# H_0(x) = 1
# H_4(x) = x^4 - 6x^2 + 3
# H_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H_12(x) = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945

def _H0(x):
    return np.ones_like(x)

def _H4(x):
    return x**4 - 6*x**2 + 3

def _H8(x):
    return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105

def _H12(x):
    return x**12 - 66*x**10 + 1155*x**8 - 6930*x**6 + 13860*x**4 - 7560*x**2 + 945

def _compute_uncertainty(a0, a4, a8):
    """Compute the uncertainty product A(f)*A(F(f)) for given coefficients."""
    if time.time() > DEADLINE:
        return None
    
    # Values at x=0
    H0_0 = 1.0
    H4_0 = 3.0  # 0 - 0 + 3
    H8_0 = 105.0  # 0 + 105
    H12_0 = 945.0  # 0 + 945
    
    # Constraint: g(0) = 0 => a0*H0_0 + a4*H4_0 + a8*H8_0 + a12*H12_0 = 0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    def g(x):
        return (a0 * _H0(x) + a4 * _H4(x) + a8 * _H8(x) + a12 * _H12(x)) * np.exp(-x**2 / 2)
    
    def g_scalar(x):
        return (a0 * 1.0 + a4 * (x**4 - 6*x**2 + 3) + 
                a8 * (x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105) +
                a12 * (x**12 - 66*x**10 + 1155*x**8 - 6930*x**6 + 13860*x**4 - 7560*x**2 + 945)) * np.exp(-x**2 / 2)
    
    # Compute ||g||^2 = integral of |g(x)|^2 dx
    norm_sq, _ = quad(lambda x: g_scalar(x)**2, -np.inf, np.inf, limit=200)
    
    if norm_sq <= 0 or not np.isfinite(norm_sq):
        return None
    
    # For Fourier-invariant functions, A(f)*A(F(f)) relates to the uncertainty product
    # The uncertainty product is (sigma_x * sigma_p)^2 where sigma_x^2 = <x^2> - <x>^2
    # and sigma_p^2 = <p^2> - <p>^2 with p = -i d/dx
    
    # <x> = integral x*|g|^2 / ||g||^2
    x_mean, _ = quad(lambda x: x * g_scalar(x)**2, -np.inf, np.inf, limit=200)
    x_mean /= norm_sq
    
    # <x^2> = integral x^2*|g|^2 / ||g||^2
    x2_mean, _ = quad(lambda x: x**2 * g_scalar(x)**2, -np.inf, np.inf, limit=200)
    x2_mean /= norm_sq
    
    sigma_x2 = x2_mean - x_mean**2
    
    # For the momentum variance, we need |g'(x)|^2
    # g'(x) = d/dx [(a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)]
    # = [a0*H0' + a4*H4' + a8*H8' + a12*H12'] * exp(-x^2/2) + (a0*H0 + a4*H4 + a8*H8 + a12*H12) * (-x) * exp(-x^2/2)
    
    def gp_scalar(x):
        H0 = 1.0
        H4 = x**4 - 6*x**2 + 3
        H8 = x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
        H12 = x**12 - 66*x**10 + 1155*x**8 - 6930*x**6 + 13860*x**4 - 7560*x**2 + 945
        
        H0p = 0.0
        H4p = 4*x**3 - 12*x
        H8p = 8*x**7 - 168*x**5 + 840*x**3 - 840*x
        H12p = 12*x**11 - 660*x**9 + 9240*x**7 - 41580*x**5 + 55440*x**3 - 15120*x
        
        poly = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
        poly_p = a0 * H0p + a4 * H4p + a8 * H8p + a12 * H12p
        
        return (poly_p - x * poly) * np.exp(-x**2 / 2)
    
    # <p^2> = integral |g'(x)|^2 / ||g||^2 (since p = -i d/dx, |p g|^2 = |g'|^2)
    p2_mean, _ = quad(lambda x: gp_scalar(x)**2, -np.inf, np.inf, limit=200)
    p2_mean /= norm_sq
    
    # <p> = integral p*|g|^2... but for real functions, <p> = -i * integral g* g' = 0 by symmetry
    # Actually for real g, <p> = 0 if g is even, but let's compute properly
    # <p> = integral (-i) g'(x) g(x) dx / ||g||^2... this is imaginary for real g
    # For the uncertainty product, we use sigma_p^2 = <p^2> - <p>^2
    # For real-valued functions, <p> is purely imaginary, so <p>^2 is negative real
    # But the standard uncertainty product uses Var(p) = <p^2> - <p>^2
    
    # For even real functions, <p> = 0, so sigma_p^2 = p2_mean
    # Let's assume the function is even (which it is since all H_n for even n are even)
    sigma_p2 = p2_mean
    
    if sigma_x2 < 0 or sigma_p2 < 0:
        return None
    
    uncertainty_product = np.sqrt(sigma_x2 * sigma_p2)
    
    if not np.isfinite(uncertainty_product):
        return None
    
    return uncertainty_product

def construct():
    """Construct the three coefficients that minimize the uncertainty product."""
    if time.time() > DEADLINE:
        return np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    best_score = float('inf')
    best_coeffs = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    # Objective function for scipy.optimize
    def objective(coeffs):
        if time.time() > DEADLINE:
            return 1e10
        a0, a4, a8 = coeffs
        result = _compute_uncertainty(a0, a4, a8)
        if result is None:
            return 1e10
        return result
    
    # Initial guesses
    initial_guesses = [
        [-113/100, 1/25, 1/3240],  # Current best
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
        [1, 1, 1],
        [-1, -1, -1],
        [1, -1, 1],
        [-1, 1, -1],
    ]
    
    for x0 in initial_guesses:
        if time.time() > DEADLINE:
            break
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                             options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.success and result.fun < best_score:
                best_score = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    # Try a few more random starts if time permits
    if time.time() < DEADLINE - 30:
        rng = np.random.default_rng(42)
        for _ in range(5):
            if time.time() > DEADLINE - 10:
                break
            x0 = rng.uniform(-5, 5, size=3)
            try:
                result = minimize(objective, x0, method='Nelder-Mead',
                                 options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
                if result.success and result.fun < best_score:
                    best_score = result.fun
                    best_coeffs = result.x
            except Exception:
                pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
