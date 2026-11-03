"""Refined Cohn-Goncalves Laguerre polynomial construction optimizer."""
import numpy as np
from scipy.optimize import minimize
import time
import math

# EVOLVE-BLOCK-START

# Precompute factorials for moment computation
# We need k! for k up to 48 (degree of Q^2 where Q is degree 24)
_FACTS = np.array([math.factorial(k) for k in range(60)], dtype=np.float64)

def _poly_moment(coeffs):
    """
    Compute integral of poly(x) * e^{-x} dx over [0, inf).
    coeffs[k] is coefficient of x^k.
    Result = sum_k coeffs[k] * k!
    """
    n = len(coeffs)
    if n == 0:
        return 0.0
    # Use dot product; factorial grows fast so for large degree this can overflow in float64
    # but for degree < 170 it is safe. Here max degree is 48.
    return np.dot(coeffs, _FACTS[:n])

def _compute_c4(roots):
    """
    Compute the C4 bound as the L2 norm of the polynomial Q(x) = prod (x - r_i)^2
    with respect to the weight e^{-x}.
    C4 = sqrt( integral_0^inf Q(x)^2 e^{-x} dx )
    """
    m = len(roots)
    roots = np.array(roots, dtype=float)
    
    # Build Q(x) = prod (x - r_i)^2
    # Start with [1]
    q_coeffs = np.array([1.0])
    
    for r in roots:
        # (x - r)^2 = x^2 - 2rx + r^2
        # Coeffs for (x-r)^2: [r^2, -2r, 1]
        factor = np.array([r*r, -2.0*r, 1.0])
        q_coeffs = np.convolve(q_coeffs, factor)
        
    # Now we have Q(x) coefficients in ascending order (constant term first)
    # We need Q(x)^2
    q2_coeffs = np.convolve(q_coeffs, q_coeffs)
    
    # Compute the moment
    integral = _poly_moment(q2_coeffs)
    
    # The bound is the square root of the integral (L2 norm)
    # Or maybe just the integral? The task says "upper bound on C4".
    # Usually norms are sqrt(integral). Let's return sqrt(integral).
    if integral < 0:
        integral = 0.0
        
    return math.sqrt(integral)

def _objective(x):
    """
    Objective function for optimization.
    x: array of 12 root positions.
    Returns: C4 bound value.
    """
    x = np.sort(x)
    if np.any(x <= 0):
        return 1e100
    if np.any(np.diff(x) <= 1e-9):
        return 1e100
    try:
        return _compute_c4(x)
    except Exception:
        return 1e100

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination
    (m=12, n=1) that minimize the C4 upper bound.
    
    Returns: list of 12 positive double-root locations
    """
    start_time = time.time()
    DEADLINE = 630  # seconds
    SAFETY_MARGIN = 30  # seconds
    
    m = 12
    
    # Initial guess: roots of Laguerre polynomial L_12(x)
    try:
        from scipy.special import roots_genlaguerre
        lag_roots, _ = roots_genlaguerre(m, 0)
        initial_guess = np.sort(lag_roots)
    except Exception:
        initial_guess = 4.0 * np.arange(1, m + 1, dtype=float)
    
    best_x = initial_guess.copy()
    best_val = _objective(best_x)
    
    # Multi-start optimization
    n_starts = 10
    np.random.seed(42)
    
    for s in range(n_starts):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - SAFETY_MARGIN:
            break
        
        if s == 0:
            x0 = initial_guess.copy()
        elif s == 1:
            # Slightly perturbed Laguerre roots
            x0 = initial_guess * (1 + 0.1 * np.random.randn(m))
            x0 = np.sort(np.abs(x0) + 0.01)
        elif s == 2:
            # Evenly spaced
            x0 = np.linspace(1.0, 50.0, m)
        elif s == 3:
            # Geometric
            x0 = np.geomspace(1.0, 50.0, m)
        else:
            # Random positive
            x0 = np.sort(1.0 + 10.0 * np.random.rand(m))
        
        try:
            res = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = np.sort(res.x)
        except Exception:
            pass
        
        # Try Nelder-Mead if time permits
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - SAFETY_MARGIN:
            break
            
        try:
            res2 = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res2.fun < best_val:
                best_val = res2.fun
                best_x = np.sort(res2.x)
        except Exception:
            pass

    # Final polish
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - SAFETY_MARGIN:
        try:
            res_final = minimize(
                _objective,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res_final.fun < best_val:
                best_val = res_final.fun
                best_x = np.sort(res_final.x)
        except Exception:
            pass
    
    # Ensure validity
    best_x = np.sort(best_x)
    if np.any(best_x <= 0):
        best_x = 4.0 * np.arange(1, m + 1, dtype=float)
    
    return list(best_x)

# EVOLVE-BLOCK-END
