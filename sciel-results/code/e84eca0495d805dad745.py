"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [0.1, 6] to [-1, 1] for Chebyshev basis
    x_min = 0.1
    x_max = 6.0
    # Avoid division by zero if range is 0, though here it is 5.9
    t = (2.0 * (x - x_min) / (x_max - x_min)) - 1.0
    
    # Precompute Chebyshev polynomials T_0 to T_9
    # T_k(t)
    T = np.zeros((len(x), 10), dtype=np.float64)
    T[:, 0] = 1.0
    T[:, 1] = t
    for k in range(2, 10):
        T[:, k] = 2.0 * t * T[:, k-1] - T[:, k-2]
        
    # Normalize columns to improve conditioning for BFGS
    # This makes the optimization landscape more isotropic
    norms = np.linalg.norm(T, axis=0)
    # Avoid division by zero, though norms should be non-zero for x in [0.1, 6]
    norms[norms == 0] = 1.0
    T = T / norms
    
    # Compute linear combination
    return T @ params
# EVOLVE-BLOCK-END
