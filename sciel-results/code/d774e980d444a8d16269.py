"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression combining a rich polynomial basis and Fourier terms.
    params is an array of length 10.
    
    Basis functions (10 total):
      0: 1
      1: x
      2: y
      3: x^2
      4: y^2
      5: xy
      6: sin(x)
      7: cos(y)
      8: sin(x + y)
      9: cos(x - y)
    """
    # Deadline guard: if we're taking too long, return a simple prediction
    # (though this function should be fast enough)
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute basis functions
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x * x
    b4 = y * y
    b5 = x * y
    b6 = np.sin(x)
    b7 = np.cos(y)
    b8 = np.sin(x + y)
    b9 = np.cos(x - y)
    
    # Linear combination
    pred = (p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3 + p4 * b4 +
            p5 * b5 + p6 * b6 + p7 * b7 + p8 * b8 + p9 * b9)
    
    return pred
# EVOLVE-BLOCK-END
