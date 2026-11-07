"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Flexible model: sum of basis functions with 10 parameters.
    Basis: 1, x, x^2, x^3, x^4, x^5, sin(x), cos(x), exp(-x^2), log1p(x)
    This provides enough flexibility to fit many common functions while
    remaining differentiable and numerically stable for BFGS.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    
    # Ensure finite input
    x_safe = np.where(np.isfinite(x), x, 1.0)
    
    p = params
    
    # Basis functions
    b0 = np.ones_like(x_safe)
    b1 = x_safe
    b2 = x_safe ** 2
    b3 = x_safe ** 3
    b4 = x_safe ** 4
    b5 = x_safe ** 5
    b6 = np.sin(x_safe)
    b7 = np.cos(x_safe)
    b8 = np.exp(-x_safe ** 2)
    b9 = np.log1p(np.abs(x_safe))  # log1p(x) but safe for all x
    
    # Linear combination
    y = (p[0] * b0 + 
         p[1] * b1 + 
         p[2] * b2 + 
         p[3] * b3 + 
         p[4] * b4 + 
         p[5] * b5 + 
         p[6] * b6 + 
         p[7] * b7 + 
         p[8] * b8 + 
         p[9] * b9)
    
    # Ensure finite output
    y = np.where(np.isfinite(y), y, 0.0)
    
    return y
# EVOLVE-BLOCK-END
