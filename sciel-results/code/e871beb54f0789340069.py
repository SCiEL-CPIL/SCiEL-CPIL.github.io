"""Rewritten program for symbolic regression with 1 input."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a Chebyshev polynomial basis.
    
    We map the input x from the domain [0.1, 6] to the standard interval
    [-1, 1] and expand the function in terms of Chebyshev polynomials T_0 to T_9.
    
    Chebyshev polynomials are well-conditioned and provide a stable basis for
    approximating smooth functions. This linear-in-parameters model allows
    BFGS to efficiently find the best coefficients.
    
    params (length 10) are the coefficients for T_0(x) through T_9(x).
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Map x from [0.1, 6] to t in [-1, 1]
    min_x = 0.1
    max_x = 6.0
    t = (2.0 * x - (min_x + max_x)) / (max_x - min_x)
    
    # Compute Chebyshev polynomials T_k(t) for k=0..9
    # T_0(t) = 1
    # T_1(t) = t
    # T_{k+1}(t) = 2*t*T_k(t) - T_{k-1}(t)
    
    T = np.zeros((10, x.shape[0]), dtype=np.float64)
    T[0] = 1.0
    T[1] = t
    
    for k in range(2, 10):
        T[k] = 2.0 * t * T[k-1] - T[k-2]
        
    # Linear combination with mild damping of higher-order terms for stability
    # This improves conditioning without changing the model class
    result = np.zeros_like(x)
    for k in range(10):
        # Damp higher-order terms slightly to improve numerical stability
        scale = 1.0 / (1.0 + 0.1 * k)
        result += params[k] * scale * T[k]
        
    return result
# EVOLVE-BLOCK-END
