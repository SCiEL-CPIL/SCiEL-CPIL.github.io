"""Symbolic regression: Chebyshev polynomial basis for smooth function approximation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate an unknown function using a Chebyshev polynomial basis of degree 9.
    
    The 10 parameters are the coefficients of T_0(x'), T_1(x'), ..., T_9(x')
    where x' = 2*(x - 0.1)/(6 - 0.1) - 1 maps x from [0.1, 6] to [-1, 1].
    
    Chebyshev polynomials provide:
    - Fast convergence for smooth functions
    - Good numerical conditioning compared to monomials
    - Linear-in-parameters form suitable for BFGS
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Map x from [0.1, 6] to [-1, 1] for Chebyshev evaluation
    x_min = 0.1
    x_max = 6.0
    x_norm = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
    
    # Compute Chebyshev polynomials T_0 through T_9 using recurrence
    # T_0(x) = 1
    # T_1(x) = x
    # T_{n+1}(x) = 2*x*T_n(x) - T_{n-1}(x)
    
    T = np.empty((10, x.shape[0]), dtype=np.float64)
    T[0, :] = 1.0
    if x.shape[0] > 0:
        T[1, :] = x_norm
        for n in range(2, 10):
            T[n, :] = 2.0 * x_norm * T[n-1, :] - T[n-2, :]
    
    # Linear combination: sum_{k=0}^{9} params[k] * T_k(x_norm)
    # params has length 10
    p = params[:10]
    prediction = T @ p
    
    return prediction
# EVOLVE-BLOCK-END
