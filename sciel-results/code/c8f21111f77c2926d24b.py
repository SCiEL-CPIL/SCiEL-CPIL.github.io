"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = params[:10]
    
    # Map x from [-3, 3] to [-1, 1] for Chebyshev polynomials
    # t = (x - 0) / 3
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2t T_{n-1}(t) - T_{n-2}(t)
    
    T = np.zeros((10, len(x)), dtype=np.float64)
    T[0, :] = 1.0
    if len(x) > 0:
        T[1, :] = t
        for n in range(2, 10):
            T[n, :] = 2.0 * t * T[n-1, :] - T[n-2, :]
            
    # Linear combination of Chebyshev polynomials
    # y = sum(p_i * T_i(t))
    # Using matrix multiplication for efficiency
    result = T @ p
    
    return result
# EVOLVE-BLOCK-END
