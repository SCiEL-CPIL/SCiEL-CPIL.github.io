"""Symbolic regression with high-order Chebyshev basis for optimal conditioning and approximation."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a basis of Chebyshev polynomials T0 to T9.
    
    Chebyshev polynomials are orthogonal on [-1, 1] and provide spectral accuracy
    for smooth functions. By mapping the input domain [-3, 3] to [-1, 1],
    we ensure the basis functions are well-conditioned and bounded in magnitude,
    which facilitates stable optimization with BFGS.
    
    Basis functions (10 total):
    1. T0(t) = 1
    2. T1(t) = t
    3. T2(t) = 2t^2 - 1
    ...
    10. T9(t)
    
    where t = x / 3.0.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    n = x.shape[0]
    
    if n == 0:
        return np.array([], dtype=np.float64)

    # Map x from [-3, 3] to [-1, 1]
    t = x / 3.0
    
    # Clip to [-1, 1] to prevent numerical overflow in polynomial recurrence
    # although the problem states x is in [-3, 3], this is a safety measure.
    t = np.clip(t, -1.0, 1.0)
    
    # Initialize result
    result = np.zeros(n, dtype=np.float64)
    
    # T0
    T_k_minus_2 = np.ones(n, dtype=np.float64)
    result += params[0] * T_k_minus_2
    
    # T1
    T_k_minus_1 = t
    result += params[1] * T_k_minus_1
    
    # Compute T2 to T9 using recurrence: T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    for k in range(2, 10):
        T_k = 2.0 * t * T_k_minus_1 - T_k_minus_2
        result += params[k] * T_k
        T_k_minus_2 = T_k_minus_1
        T_k_minus_1 = T_k
        
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.zeros_like(x)
        
    return result
# EVOLVE-BLOCK-END
