"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate target function using Chebyshev polynomial basis on [-2, 2].
    Maps x to t in [-1, 1] and computes T_0(t) through T_9(t).
    Returns linear combination: sum_{k=0}^{9} params[k] * T_k(x/2)
    """
    # Sanitize params to avoid non-finite values during optimization
    if not np.all(np.isfinite(params)):
        params = np.ones(10)
    
    # Map x from [-2, 2] to t in [-1, 1]
    t = x / 2.0
    
    N = x.shape[0]
    if N == 0:
        return np.array([])
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    
    # Use a stable iterative computation
    # result = params[0] * T_0 + params[1] * T_1 + ...
    
    T_prev2 = np.ones(N)  # T_0
    T_prev1 = t.copy()     # T_1
    
    result = params[0] * T_prev2 + params[1] * T_prev1
    
    for k in range(2, 10):
        T_curr = 2.0 * t * T_prev1 - T_prev2
        result = result + params[k] * T_curr
        T_prev2 = T_prev1
        T_prev1 = T_curr
    
    return result
# EVOLVE-BLOCK-END
