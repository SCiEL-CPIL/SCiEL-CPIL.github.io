"""Symbolic regression with Chebyshev polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Compute y = sum_{i=0}^{9} params[i] * T_i(x/3)
    
    Uses Chebyshev polynomials on the scaled domain [-1, 1] 
    to improve conditioning and approximation properties.
    Chebyshev polynomials are orthogonal on [-1, 1] and provide
    a well-conditioned basis for approximating smooth functions.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure we have at least 10 params
    if p.shape[0] < 10:
        p = np.pad(p, (0, 10 - p.shape[0]), constant_values=0.0)
    
    # Scale x to [-1, 1]
    z = x / 3.0
    
    # Compute Chebyshev polynomials T_0 ... T_9
    # T_0(z) = 1
    # T_1(z) = z
    # T_k(z) = 2*z*T_{k-1}(z) - T_{k-2}(z)
    
    T_prev2 = np.ones_like(z)  # T_0
    T_prev1 = z.copy()          # T_1
    
    result = p[0] * T_prev2
    if len(p) > 1:
        result += p[1] * T_prev1
        
    for k in range(2, 10):
        T_curr = 2.0 * z * T_prev1 - T_prev2
        result += p[k] * T_curr
        T_prev2 = T_prev1
        T_prev1 = T_curr
        
    return result
# EVOLVE-BLOCK-END
