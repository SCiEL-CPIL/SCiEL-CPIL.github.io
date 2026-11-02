"""Structural change: Use the specific L4-norm product objective derived from Hermite Fourier properties."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """Return three coefficients for H0, H4, H8 Hermite combination."""
    # H12 coefficient is fixed by g(0) = 0:
    # H0(0)=1, H4(0)=-3, H8(0)=105, H12(0)=-10395
    # a12 = (a0 - 3*a4 + 105*a8) / 10395
    
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin
    
    # Pre-compute grid for integration
    # The integrand is |P(x)|^4 * exp(-2x^2)
    # This decays very fast. [-10, 10] is sufficient.
    x = np.linspace(-10, 10, 20000)
    dx = x[1] - x[0]
    gauss_weight = np.exp(-2.0 * x**2)
    
    # Pre-compute Hermite polynomials on the grid
    H0 = np.ones_like(x)
    H4 = 16 * x**4 - 48 * x**2 + 12
    H8 = 256 * x**8 - 2304 * x**6 + 5760 * x**4 - 4320 * x**2 + 504
    H12 = 4096 * x**12 - 55296 * x**10 + 303104 * x**8 - 589824 * x**6 + 408192 * x**4 - 77760 * x**2 + 10395
    
    def objective(coeffs):
        a0, a4, a8 = coeffs
        a12 = (a0 - 3.0 * a4 + 105.0 * a8) / 10395.0
        
        # P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
        P = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
        
        # Objective: Integral of |P(x)|^4 * exp(-2x^2) dx
        # This is proportional to ||g||_4^4 * ||F(g)||_4^4
        integrand = np.abs(P)**4 * gauss_weight
        val = np.sum(integrand) * dx
        
        return val

    best_coeffs = seed.copy()
    best_val = objective(seed)
    
    # Perform a local optimization starting from the seed
    # The seed is likely very close to the optimum.
    # Use BFGS or L-BFGS-B for efficiency.
    try:
        # Check time
        if time.time() - start_time < deadline:
            result = minimize(
                objective,
                seed,
                method='BFGS',
                options={'maxiter': 1000, 'gtol': 1e-8}
            )
            if result.fun < best_val:
                best_coeffs = result.x
                best_val = result.fun
    except Exception:
        pass
    
    # Try a second pass with Nelder-Mead if time permits, just in case BFGS got stuck
    if time.time() - start_time < deadline:
        try:
            result2 = minimize(
                objective,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if result2.fun < best_val:
                best_coeffs = result2.x
                best_val = result2.fun
        except Exception:
            pass
            
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
