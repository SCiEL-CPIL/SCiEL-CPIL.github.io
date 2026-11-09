"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of well-conditioned basis functions for symbolic regression.
    Uses orthogonal-ish basis to help BFGS converge from all-ones start.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Build a diverse set of basis functions that are less correlated
    # and cover common functional forms in symbolic regression
    
    # b0: constant
    b0 = np.ones_like(x)
    
    # b1: centered linear (reduce correlation with constant)
    b1 = (x - 3.05) / 3.05  # centered at midpoint of [0.1, 6]
    
    # b2: quadratic centered
    x_c = (x - 3.05) / 3.05
    b2 = x_c * x_c
    
    # b3: cubic centered
    b3 = x_c * x_c * x_c
    
    # b4: sin(x) - covers periodic behavior
    b4 = np.sin(x)
    
    # b5: cos(x) - complementary periodic
    b5 = np.cos(x)
    
    # b6: exp(x) scaled to be O(1)
    b6 = np.exp(x - 3.05) / np.exp(3.05)  # normalize so exp(3.05) maps to 1
    
    # b7: exp(-x) scaled
    b7 = np.exp(-x) * np.exp(3.05)  # so exp(-3.05) maps to 1
    
    # b8: log(x) centered
    b8 = np.log(x) - np.log(3.05)
    
    # b9: 1/x centered and scaled
    b9 = (1.0 / x - 1.0 / 3.05) * 3.05
    
    # Stack into matrix and compute linear combination
    basis = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    y = basis @ params
    
    return y

# EVOLVE-BLOCK-END
