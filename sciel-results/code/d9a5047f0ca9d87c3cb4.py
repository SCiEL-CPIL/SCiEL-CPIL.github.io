"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale x to [-1, 1] to improve numerical stability
    u = np.asarray(x, dtype=np.float64) / 3.0
    
    # Compute Chebyshev polynomials of the first kind T_0 to T_9
    # T_0(u) = 1
    # T_1(u) = u
    # T_k(u) = 2*u*T_{k-1}(u) - T_{k-2}(u)
    
    # Initialize basis matrix
    # We have 10 parameters, so we need basis functions up to degree 9
    # basis[i] will hold T_i(u)
    
    T0 = np.ones_like(u)
    T1 = u.copy()
    
    # Construct the linear combination
    # result = p[0]*T0 + p[1]*T1 + ... + p[9]*T9
    result = params[0] * T0 + params[1] * T1
    
    if len(params) > 2:
        T_prev2 = T0
        T_prev1 = T1
        for k in range(2, 10):
            # T_k = 2*u*T_{k-1} - T_{k-2}
            T_curr = 2.0 * u * T_prev1 - T_prev2
            result += params[k] * T_curr
            T_prev2 = T_prev1
            T_prev1 = T_curr

    return result
# EVOLVE-BLOCK-END
