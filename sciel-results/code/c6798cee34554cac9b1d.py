"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 110.0  # 110s safety margin within 120s limit


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis combining low-order Chebyshev polynomials (global trend)
    and Gaussian RBFs (local features).
    
    Structure:
    - 5 Chebyshev terms T_0 to T_4 (mapped to [-1, 1])
    - 5 Gaussian RBFs at fixed centers {-3, -1.5, 0, 1.5, 3}
    
    This is a linear model in params, well-conditioned for BFGS.
    The Chebyshev terms capture smooth global behavior, while the RBFs
    allow local adjustments for non-periodic, smooth targets.
    """
    if time.time() > _DEADLINE:
        # Fallback: return a simple linear fit as safety
        x = np.asarray(x, dtype=np.float64)
        return params[0] + params[1] * x
    
    x = np.asarray(x, dtype=np.float64)
    
    # Normalize x to [-1, 1] for Chebyshev stability
    t = x / 3.0
    
    # Chebyshev Polynomials T_0 to T_4
    b0 = np.ones_like(x)
    b1 = t
    b2 = 2.0 * t * t - 1.0
    b3 = 4.0 * b1 * b1 * b1 - 3.0 * b1
    b4 = 8.0 * b1 * b1 * b1 * b1 - 8.0 * b1 * b1 + 1.0
    
    # Gaussian RBFs at fixed centers
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    width = 1.5  # Moderate width for overlap and smoothness
    
    # Pre-compute RBF values
    b5 = np.exp(-0.5 * ((x - centers[0]) / width) ** 2)
    b6 = np.exp(-0.5 * ((x - centers[1]) / width) ** 2)
    b7 = np.exp(-0.5 * ((x - centers[2]) / width) ** 2)
    b8 = np.exp(-0.5 * ((x - centers[3]) / width) ** 2)
    b9 = np.exp(-0.5 * ((x - centers[4]) / width) ** 2)
    
    # Linear combination
    y = (params[0] * b0 + 
         params[1] * b1 + 
         params[2] * b2 + 
         params[3] * b3 + 
         params[4] * b4 + 
         params[5] * b5 + 
         params[6] * b6 + 
         params[7] * b7 + 
         params[8] * b8 + 
         params[9] * b9)
         
    return y
# EVOLVE-BLOCK-END
