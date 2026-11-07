"""
Symbolic regression using Chebyshev polynomial basis.
The hidden function is approximated by a linear combination of Chebyshev polynomials
of the first kind, scaled to the domain [-1, 1].
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    x: array of input points
    params: array of 10 coefficients
    Returns: predicted values
    """
    # Scale x from [-2, 2] to [-1, 1]
    # This aligns with the natural domain of Chebyshev polynomials
    t = 0.5 * x
    
    # Compute Chebyshev polynomials T_0(t) ... T_9(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    
    # Initialize the basis matrix
    # We need 10 basis functions corresponding to params[0..9]
    n_params = 10
    basis = np.zeros((x.size, n_params))
    
    if x.size == 0:
        return np.array([])

    # T_0
    basis[:, 0] = 1.0
    # T_1
    basis[:, 1] = t
    
    # Recurrence relation for higher orders
    T_prev2 = basis[:, 0]
    T_prev1 = basis[:, 1]
    
    for k in range(2, n_params):
        T_curr = 2.0 * t * T_prev1 - T_prev2
        basis[:, k] = T_curr
        T_prev2 = T_prev1
        T_prev1 = T_curr
        
    # Linear combination
    return basis @ params
# EVOLVE-BLOCK-END
