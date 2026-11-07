"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a linear combination of fixed basis functions.
    The model is linear in params, making BFGS convergence reliable.
    
    Parameters (length 10):
    p0: constant
    p1: x
    p2: x^2
    p3: x^3
    p4: x^4
    p5: sin(x)
    p6: cos(x)
    p7: tanh(x)
    p8: exp(x)
    p9: exp(-x)
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Fixed basis functions on x in [-2, 2]
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    y = (p0 * 1.0
         + p1 * x
         + p2 * x2
         + p3 * x3
         + p4 * x4
         + p5 * np.sin(x)
         + p6 * np.cos(x)
         + p7 * np.tanh(x)
         + p8 * np.exp(x)
         + p9 * np.exp(-x))
    
    return y
# EVOLVE-BLOCK-END
