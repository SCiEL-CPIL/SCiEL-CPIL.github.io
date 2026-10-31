import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_hermite
from scipy.integrate import quad

def construct():
    """
    Optimize the three free Hermite coefficients (H0, H4, H8) to minimize
    the uncertainty bound, with H12 determined by g(0)=0.
    """
    # Hermite polynomial values at 0 for n = 0, 4, 8, 12
    H0_0 = 1.0
    H4_0 = 3.0
    H8_0 = 105.0
    H12_0 = 10395.0
    
    def get_all_coeffs(a0, a4, a8):
        """Return all four coefficients with a12 determined by g(0)=0."""
        a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
        return np.array([a0, a4, a8, a12])
    
    def g_func(x, coeffs):
        """Evaluate g(x) = sum a_n H_n(x) exp(-x^2/2)"""
        n_vals = [0, 4, 8, 12]
        result = np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0
        for n, a in zip(n_vals, coeffs):
            result += a * eval_hermite(n, x) * np.exp(-x**2 / 2)
        return result
    
    def g_hat_func(xi, coeffs):
        """
        Fourier transform of g. 
        F[H_n(x) e^{-x^2/2}](xi) = i^n H_n(xi) e^{-xi^2/2}
        """
        n_vals = [0, 4, 8, 12]
        result = np.zeros_like(xi) if isinstance(xi, np.ndarray) else 0.0
        for n, a in zip(n_vals, coeffs):
            result += a * (1j)**n * eval_hermite(n, xi) * np.exp(-xi**2 / 2)
        return result
    
    def compute_l1_norms(coeffs):
        """Compute ||g||_1 and ||g_hat||_1 via numerical quadrature."""
        def abs_g(x):
            return np.abs(g_func(x, coeffs))
        
        def abs_g_hat(xi):
            return np.abs(g_hat_func(xi, coeffs))
        
        # Use a wider range and higher limit for accuracy
        l1_g, _ = quad(abs_g, -15, 15, limit=300, epsabs=1e-10, epsrel=1e-10)
        l1_g_hat, _ = quad(abs_g_hat, -15, 15, limit=300, epsabs=1e-10, epsrel=1e-10)
        
        return l1_g, l1_g_hat
    
    def objective(params):
        """Objective: minimize the product of L1 norms."""
        a0, a4, a8 = params
        coeffs = get_all_coeffs(a0, a4, a8)
        
        try:
            l1_g, l1_g_hat = compute_l1_norms(coeffs)
        except Exception:
            return 1e10
            
        if l1_g < 1e-12 or l1_g_hat < 1e-12:
            return 1e10
            
        return l1_g * l1_g_hat

    # Seed values
    x0 = np.array([-113/100, 1/25, 1/3240])
    
    # Evaluate seed first to have a baseline
    seed_value = objective(x0)
    
    # Strategy:
    # 1. Try a global optimization (differential_evolution) to find a better spot.
    # 2. Refine with local minimization.
    # 3. If any result is worse than seed, use seed.
    
    best_params = x0
    best_value = seed_value
    
    # Global Search
    try:
        bounds = [(-10, 10), (-1, 1), (-0.1, 0.1)]
        # Differential evolution is robust but can be slow. 
        # We limit maxiter to ensure we finish in time.
        result_global = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=1e-6,
            seed=42,
            polish=False
        )
        if result_global.fun < best_value:
            best_value = result_global.fun
            best_params = result_global.x
    except Exception:
        pass
    
    # Local Refinement from the best found so far
    try:
        result_local = minimize(
            objective,
            best_params,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        if result_local.fun < best_value:
            best_value = result_local.fun
            best_params = result_local.x
    except Exception:
        pass
        
    # Final check: if the best found is worse than seed, revert to seed
    if best_value > seed_value:
        best_params = x0
        best_value = seed_value
        
    return np.array(best_params)
