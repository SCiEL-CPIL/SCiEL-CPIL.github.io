"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fits a Chebyshev polynomial series of degree 9 to the data.
    Input x is in [-2, 2]. We scale to t in [-1, 1] for Chebyshev stability.
    params contains 10 coefficients for T_0(t) through T_9(t).
    """
    # Scale x from [-2, 2] to [-1, 1]
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0(t) to T_9(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    # We can compute these iteratively and accumulate the result
    # This avoids storing a large matrix and is efficient.
    
    # Initialize basis vectors
    # We need to return a linear combination: sum(params[i] * T_i(t))
    
    # T_0
    T_0 = np.ones_like(t)
    # T_1
    T_1 = t.copy()
    
    result = params[0] * T_0 + params[1] * T_1
    
    if len(params) > 2:
        T_prev = T_0
        T_curr = T_1
        for k in range(2, len(params)):
            # T_k = 2*t*T_{k-1} - T_{k-2}
            T_next = 2.0 * t * T_curr - T_prev
            result += params[k] * T_next
            T_prev = T_curr
            T_curr = T_next
            
    return result
# EVOLVE-BLOCK-END
