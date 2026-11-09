"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale input x from [-2, 2] to [-1, 1] for Chebyshev polynomials
    t = x / 2.0
    
    # Initialize basis matrix
    # We use 10 parameters, so we compute T_0 to T_9
    n_points = len(x)
    basis = np.zeros((n_points, 10))
    
    if n_points == 0:
        return np.array([])
        
    # T_0(t) = 1
    basis[:, 0] = 1.0
    
    # T_1(t) = t
    basis[:, 1] = t
    
    # Recurrence: T_{k+1}(t) = 2*t*T_k(t) - T_{k-1}(t)
    for k in range(2, 10):
        basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
        
    # Linear combination
    return basis @ params
# EVOLVE-BLOCK-END
