"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _chebyshev_basis(x_mapped: np.ndarray) -> np.ndarray:
    """Compute Chebyshev polynomials T_0..T_4 for x in [-1, 1]."""
    # x_mapped should be in [-1, 1]
    t0 = np.ones_like(x_mapped)
    t1 = x_mapped
    t2 = 2.0 * x_mapped * x_mapped - 1.0
    t3 = 4.0 * x_mapped**3 - 3.0 * x_mapped
    t4 = 8.0 * x_mapped**4 - 8.0 * x_mapped**2 + 1.0
    return np.column_stack([t0, t1, t2, t3, t4])

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: Chebyshev(0-4) + sin(w*x) + cos(w*x) + a*exp(b*x)
    params[0..4]: Chebyshev coefficients
    params[5]: frequency w
    params[6]: sin amplitude
    params[7]: cos amplitude
    params[8]: exp rate b
    params[9]: exp amplitude a
    """
    # Map x from [-3, 3] to [-1, 1] for Chebyshev
    x_mapped = x / 3.0
    
    # Chebyshev basis
    T = _chebyshev_basis(x_mapped)
    cheb_part = T @ params[0:5]
    
    # Sinusoidal component
    w = params[5]
    sin_part = params[6] * np.sin(w * x)
    cos_part = params[7] * np.cos(w * x)
    
    # Exponential component with clipping for stability
    b = params[8]
    a = params[9]
    exp_arg = np.clip(b * x, -50.0, 50.0)
    exp_part = a * np.exp(exp_arg)
    
    result = cheb_part + sin_part + cos_part + exp_part
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
