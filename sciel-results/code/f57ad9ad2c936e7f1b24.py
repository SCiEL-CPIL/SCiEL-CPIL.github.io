"""Combined approach: L4-norm product objective with multi-start and refined local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import differential_evolution


def construct():
    """Return three coefficients for H0, H4, H8 Hermite combination."""
    # H12 coefficient is fixed by g(0) = 0:
    # H0(0)=1, H4(0)=-3, H8(0)=105, H12(0)=-10395
    # a12 = (a0 - 3*a4 + 105*a8) / 10395
    
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    start_time = time.time()
    deadline = start_time + 175.0  # Safety margin for 210s wall-clock limit
    
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
        """Compute the L4-norm product proxy: integral of |P(x)|^4 * exp(-2x^2) dx.
        
        This is proportional to ||g||_4^4 * ||F(g)||_4^4 by Parseval and
        the Fourier properties of Hermite-Gaussian functions.
        """
        a0, a4, a8 = coeffs
        a12 = (a0 - 3.0 * a4 + 105.0 * a8) / 10395.0
        
        # P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
        P = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
        
        # Objective: Integral of |P(x)|^4 * exp(-2x^2) dx
        integrand = np.abs(P)**4 * gauss_weight
        val = np.sum(integrand) * dx
        
        return val
    
    best_coeffs = seed.copy()
    best_val = objective(seed)
    
    # Phase 1: Broad search with differential evolution (if time permits)
    if time.time() - start_time < deadline - 60.0:
        try:
            bounds = [
                (-2.0, 2.0),    # a0
                (-0.5, 0.5),    # a4
                (-0.01, 0.01),  # a8
            ]
            
            result = differential_evolution(
                objective,
                bounds=bounds,
                seed=42,
                maxiter=30,
                tol=1e-6,
                popsize=10,
                time_limit=min(60.0, deadline - 60.0 - (time.time() - start_time))
            )
            
            if result.fun < best_val:
                best_coeffs = result.x
                best_val = result.fun
        except Exception:
            pass
    
    # Phase 2: Local refinement from multiple starting points
    start_points = [best_coeffs, seed]
    
    # Add a few random perturbations around the best found so far
    for _ in range(5):
        if time.time() - start_time > deadline - 30.0:
            break
        perturbation = np.random.normal(0, 0.1, size=3)
        start_points.append(best_coeffs + perturbation)
    
    for sp in start_points:
        if time.time() - start_time > deadline - 15.0:
            break
        try:
            # Try BFGS first (faster convergence)
            result = minimize(
                objective,
                sp,
                method='BFGS',
                options={'maxiter': 500, 'gtol': 1e-8}
            )
            if result.fun < best_val:
                best_coeffs = result.x
                best_val = result.fun
            
            # Follow up with Nelder-Mead for potential further improvement
            if time.time() - start_time > deadline - 10.0:
                break
            result2 = minimize(
                objective,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if result2.fun < best_val:
                best_coeffs = result2.x
                best_val = result2.fun
        except Exception:
            pass
    
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
