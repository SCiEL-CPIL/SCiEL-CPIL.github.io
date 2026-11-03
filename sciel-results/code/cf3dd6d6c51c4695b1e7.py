"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hermite(n, x):
    """Compute physicists' Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    H_prev = np.ones_like(x)
    H_curr = 2 * x
    for k in range(2, n + 1):
        H_next = 2 * x * H_curr - 2 * (k - 1) * H_prev
        H_prev, H_curr = H_curr, H_next
    return H_curr

def _compute_c12(c0, c4, c8):
    """c12 such that g(0) = 0."""
    return -(c0 + 3.0 * c4 + 10.0 * c8) / 105.0

def _upper_bound_C4(c0, c4, c8):
    """Compute the upper bound C4."""
    c12 = _compute_c12(c0, c4, c8)
    
    # Compute ||g||_1 = ∫|P(x) e^{-x^2/2}| dx
    x = np.linspace(-15, 15, 200000)
    P = (c0 * _hermite(0, x) + 
         c4 * _hermite(4, x) + 
         c8 * _hermite(8, x) + 
         c12 * _hermite(12, x))
    g = P * np.exp(-x**2 / 2.0)
    dx = x[1] - x[0]
    L1 = np.sum(np.abs(g)) * dx
    
    # Also compute ||g||_2 for potential use
    L2 = np.sqrt(np.sum(g**2) * dx)
    
    # The bound is the L1 norm of g
    return L1

def construct():
    deadline = time.time() + 200  # 10s safety margin
    
    # Prior state of the art
    best = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_val = _upper_bound_C4(*best)
    
    # Nelder-Mead optimization from multiple starts
    np.random.seed(42)
    
    for attempt in range(30):
        if time.time() > deadline:
            break
        
        if attempt == 0:
            x0 = best.copy()
        elif attempt == 1:
            x0 = np.array([0.0, 0.0, 0.0])
        elif attempt == 2:
            x0 = np.array([-1.0, 0.05, 0.001])
        else:
            x0 = best + np.random.randn(3) * np.array([0.5, 0.05, 0.005])
        
        try:
            result = minimize(_upper_bound_C4, x0, method='Nelder-Mead',
                            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14,
                                    'adaptive': True})
            if result.fun < best_val:
                best_val = result.fun
                best = result.x.copy()
        except Exception:
            pass
    
    return best
# EVOLVE-BLOCK-END
