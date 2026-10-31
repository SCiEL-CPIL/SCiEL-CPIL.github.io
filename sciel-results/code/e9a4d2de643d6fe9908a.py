"""Program for the uncertainty inequality task using Remez-like optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import hermite
import math
from scipy.optimize import differential_evolution, minimize


def _hermite_poly(n, x):
    """Compute the nth physicist's Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    elif n == 2:
        return 4 * x**2 - 2
    elif n == 3:
        return 8 * x**3 - 12 * x
    elif n == 4:
        return 16 * x**4 - 48 * x**2 + 12
    elif n == 5:
        return 32 * x**5 - 160 * x**3 + 120 * x
    elif n == 6:
        return 64 * x**6 - 480 * x**4 + 720 * x**2 - 120
    elif n == 7:
        return 128 * x**7 - 1344 * x**5 + 3360 * x**3 - 1680 * x
    elif n == 8:
        return 256 * x**8 - 3584 * x**6 + 13440 * x**4 - 13440 * x**2 + 1680
    elif n == 12:
        # H_12(x) = 4096 x^12 - 122880 x^10 + 1290240 x^8 - 6451200 x^6 + 13107200 x^4 - 9031680 x^2 + 1032192
        return (4096 * x**12 - 122880 * x**10 + 1290240 * x**8 
                - 6451200 * x**6 + 13107200 * x**4 - 9031680 * x**2 + 1032192)
    else:
        # Fallback using scipy
        return hermite(n)(x)


def _compute_bound(c0, c4, c8):
    """
    Compute the upper bound C4 on A(f)*A(F) for the given Hermite coefficients.
    """
    c12 = -(c0 + 12 * c4 + 1680 * c8) / 1032192.0
    
    # Use a dense grid for accurate supremum estimation
    # The functions decay as x^12 * exp(-x^2/2), so range [-10, 10] is sufficient
    # 20001 points gives high precision
    x = np.linspace(-10, 10, 20001)
    
    # Pre-compute Hermite polynomials on the grid for speed
    H0 = np.ones_like(x)
    H4 = 16 * x**4 - 48 * x**2 + 12
    H8 = 256 * x**8 - 3584 * x**6 + 13440 * x**4 - 13440 * x**2 + 1680
    H12 = (4096 * x**12 - 122880 * x**10 + 1290240 * x**8 
           - 6451200 * x**6 + 13107200 * x**4 - 9031680 * x**2 + 1032192)
    
    gauss = np.exp(-x**2 / 2)
    
    # Position space function
    g = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * gauss
    A_f = np.max(np.abs(x * g))
    
    # Momentum space function (Fourier Transform)
    # FT is symmetric for these even terms
    xi = x
    sqrt_2pi = math.sqrt(2 * math.pi)
    G = sqrt_2pi * (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * gauss
    A_F = np.max(np.abs(xi * G))
    
    return A_f * A_F


def _objective(c):
    """Objective function for scipy.optimize."""
    return _compute_bound(c[0], c[1], c[2])

def _remez_optimize(c_init, max_iter=100, tol=1e-12):
    """
    Optimizes the coefficients using a combination of global search (Differential Evolution)
    and local refinement (L-BFGS-B). This is more robust than the previous heuristic Remez.
    """
    # 1. Global Search using Differential Evolution
    # Bounds for c0, c4, c8
    # Based on the known solution ~[-1.13, 0.04, 0.0003], we can set reasonable bounds
    bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.01, 0.01)]
    
    try:
        result_de = differential_evolution(
            _objective, 
            bounds=bounds, 
            seed=42, 
            maxiter=50, 
            popsize=15,
            tol=1e-8
        )
        best_c = result_de.x
        best_bound = result_de.fun
    except Exception as e:
        print(f"DE failed: {e}")
        best_c = np.array(c_init)
        best_bound = _compute_bound(*best_c)

    # 2. Local Refinement using L-BFGS-B
    try:
        result_min = minimize(
            _objective, 
            best_c, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        if result_min.fun < best_bound:
            best_c = result_min.x
            best_bound = result_min.fun
    except Exception as e:
        print(f"Minimize failed: {e}")

    return best_c, best_bound


def construct():
    """
    Return three coefficients [c0, c4, c8] that minimize the upper bound C4.
    """
    # Use the robust optimizer which includes global search
    # The initial point is used as a fallback if DE fails, but DE is the primary driver
    init_c = np.array([-1.13, 0.04, 0.0003])
    
    try:
        best_c, best_bound = _remez_optimize(init_c)
    except Exception:
        best_c = init_c
        best_bound = _compute_bound(*best_c)
        
    return np.array([best_c[0], best_c[1], best_c[2]])
# EVOLVE-BLOCK-END
