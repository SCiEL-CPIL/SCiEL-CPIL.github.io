"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Chebyshev polynomial series.
    The basis functions T_k(x/2) for k=0..9 are orthogonal on [-1, 1],
    corresponding to x in [-2, 2]. This provides a well-conditioned basis
    for approximating smooth functions, improving BFGS convergence compared
    to raw polynomials or mixed transcendental functions.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Map x from [-2, 2] to t in [-1, 1]
    t = x * 0.5
    
    # Precompute powers of t for efficiency
    t2 = t * t
    t3 = t2 * t
    t4 = t2 * t2
    t5 = t4 * t
    t6 = t4 * t2
    t7 = t6 * t
    t8 = t4 * t4
    t9 = t8 * t
    
    # Chebyshev polynomials T_0 to T_9
    b0 = np.ones_like(t)
    b1 = t
    b2 = 2.0 * t2 - 1.0
    b3 = 4.0 * t3 - 3.0 * t
    b4 = 8.0 * t4 - 8.0 * t2 + 1.0
    b5 = 16.0 * t5 - 20.0 * t3 + 5.0 * t
    b6 = 32.0 * t6 - 48.0 * t4 + 18.0 * t2 - 1.0
    b7 = 64.0 * t7 - 112.0 * t5 + 56.0 * t3 - 7.0 * t
    b8 = 128.0 * t8 - 256.0 * t6 + 160.0 * t4 - 32.0 * t2 + 1.0
    b9 = 256.0 * t9 - 576.0 * t7 + 432.0 * t5 - 120.0 * t3 + 9.0 * t
    
    p = params
    
    result = (p[0] * b0 + 
              p[1] * b1 + 
              p[2] * b2 + 
              p[3] * b3 + 
              p[4] * b4 + 
              p[5] * b5 + 
              p[6] * b6 + 
              p[7] * b7 + 
              p[8] * b8 + 
              p[9] * b9)
              
    return result
# EVOLVE-BLOCK-END
