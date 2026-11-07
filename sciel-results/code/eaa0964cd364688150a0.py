"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [-2, 2] to [-1, 1] for Chebyshev stability
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    
    T = np.zeros((x.size, 10))
    T[:, 0] = 1.0
    if x.size > 0:
        T[:, 1] = t
        for k in range(2, 10):
            T[:, k] = 2.0 * t * T[:, k-1] - T[:, k-2]
            
    return T @ params
# EVOLVE-BLOCK-END
