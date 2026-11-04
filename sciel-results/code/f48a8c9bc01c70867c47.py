import numpy as np
from scipy.optimize import minimize
import time

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    start_time = time.time()
    deadline = start_time + 600
    
    def compute_c4(roots):
        """Compute the C4 upper bound given 12 double-root locations."""
        roots = np.sort(np.asarray(roots, dtype=float))
        if np.any(roots <= 0):
            return 1e15
        
        # f(x) = exp(-x^2/2) * prod_{i=1}^{12} (x - r_i)^2
        # Use numerical integration on a suitable grid
        x = np.linspace(-80, 80, 160001)
        dx = x[1] - x[0]
        
        # Compute P(x) = prod (x - r_i)^2
        P = np.ones_like(x)
        for r in roots:
            P *= (x - r)**2
        
        f = np.exp(-x**2 / 2.0) * P
        f2 = f**2
        
        norm_f_sq = np.trapz(f2, x)
        norm_xf_sq = np.trapz(x**2 * f2, x)
        
        if norm_f_sq <= 0:
            return 1e15
            
        # Uncertainty product: ||x f||_2 * ||f̂||_2
        # By Parseval, ||f̂||_2 = ||f||_2 (with standard normalization)
        c4 = np.sqrt(norm_xf_sq * norm_f_sq)
        
        return c4
    
    # Initial guess
    x0 = np.linspace(0.5, 12.0, 12)
    
    best_roots = x0.copy()
    best_c4 = compute_c4(best_roots)
    
    # Nelder-Mead
    try:
        if time.time() < deadline - 120:
            res = minimize(lambda r: compute_c4(r), x0, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-7})
            if res.fun < best_c4:
                best_c4 = res.fun
                best_roots = np.sort(res.x)
    except Exception:
        pass
    
    # L-BFGS-B
    try:
        if time.time() < deadline - 60:
            bounds = [(0.01, 40.0)] * 12
            res2 = minimize(lambda r: compute_c4(r), x0, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 1000})
            if res2.fun < best_c4:
                best_c4 = res2.fun
                best_roots = np.sort(res2.x)
    except Exception:
        pass
    
    best_roots = np.sort(best_roots)
    best_roots = np.maximum(best_roots, 0.01)
    
    return list(best_roots)
