import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite
from scipy.integrate import quad

def construct():
    """
    Optimize the three free Hermite coefficients (H0, H4, H8) to minimize
    the normalized uncertainty bound, with H12 determined by g(0)=0.
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
        Fourier transform of g. Using convention where
        F[H_n(x) e^{-x^2/2}](xi) = (i)^n H_n(xi) e^{-xi^2/2}
        """
        n_vals = [0, 4, 8, 12]
        result = np.zeros_like(xi) if isinstance(xi, np.ndarray) else 0.0
        for n, a in zip(n_vals, coeffs):
            result += a * (1j)**n * eval_hermite(n, xi) * np.exp(-xi**2 / 2)
        return result
        
    def compute_norms(coeffs):
        """Compute ||g||_1, ||g_hat||_1, and ||g||_2 via numerical quadrature."""
        def abs_g(x):
            return np.abs(g_func(x, coeffs))
        
        def abs_g_hat(xi):
            return np.abs(g_hat_func(xi, coeffs))
            
        def g_sq(x):
            val = g_func(x, coeffs)
            return np.abs(val)**2
            
        def g_hat_sq(xi):
            val = g_hat_func(xi, coeffs)
            return np.abs(val)**2

        l1_g, _ = quad(lambda x: float(abs_g(x)), -15, 15, limit=500)
        l1_g_hat, _ = quad(lambda xi: float(abs_g_hat(xi)), -15, 15, limit=500)
        
        # L2 norm squared is integral of |g|^2
        l2_sq_g, _ = quad(lambda x: float(g_sq(x)), -15, 15, limit=500)
        
        return l1_g, l1_g_hat, l2_sq_g

    def objective(params):
        """Objective: minimize the normalized uncertainty product."""
        a0, a4, a8 = params
        coeffs = get_all_coeffs(a0, a4, a8)
        
        # Check if coefficients are effectively zero
        if np.allclose(coeffs, 0, atol=1e-12):
            return 1e6
            
        l1_g, l1_g_hat, l2_sq_g = compute_norms(coeffs)
        
        if l2_sq_g < 1e-12:
            return 1e6
            
        # The normalized uncertainty bound is proportional to (L1_g * L1_g_hat) / L2_sq_g
        return (l1_g * l1_g_hat) / l2_sq_g

    # Use multiple random restarts with Nelder-Mead to find a good local minimum
    best_params = None
    best_obj = np.inf
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    for restart in range(20):
        if restart == 0:
            x0 = np.array([-1.0, 0.1, 0.01])
        else:
            x0 = np.random.uniform(-5, 5, size=3)
        
        try:
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            
            if result.fun < best_obj and result.fun < 1e6:
                best_obj = result.fun
                best_params = result.x
                
        except Exception:
            pass
    
    if best_params is None:
        # Fallback to a reasonable guess if all optimizations fail
        best_params = np.array([-1.0, 0.1, 0.01])
        
    return np.array(best_params)
