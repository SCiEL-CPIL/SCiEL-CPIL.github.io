"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the hidden function using a Chebyshev polynomial basis of degree 9.
    Chebyshev polynomials are orthogonal on [-1, 1], providing a well-conditioned
    basis for BFGS optimization.
    """
    x = np.asarray(x, dtype=float)
    
    # Scale x from [0.1, 6] to [-1, 1]
    # Center = (0.1 + 6) / 2 = 3.05
    # Half-Range = (6 - 0.1) / 2 = 2.95
    u = (x - 3.05) / 2.95
    
    # Initialize output
    y = np.zeros_like(x)
    
    # T_0(u) = 1
    T_0 = np.ones_like(x)
    y += params[0] * T_0
    
    if len(params) > 1:
        # T_1(u) = u
        T_1 = u
        y += params[1] * T_1
        
        # Recursively compute T_2 to T_9
        # T_n(u) = 2*u*T_{n-1}(u) - T_{n-2}(u)
        for i in range(2, 10):
            T_n = 2.0 * u * T_1 - T_0
            y += params[i] * T_n
            T_0 = T_1
            T_1 = T_n
            
    return y
# EVOLVE-BLOCK-END
