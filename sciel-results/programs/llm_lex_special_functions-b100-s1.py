"""Symbolic regression with 1 input; Chebyshev polynomial basis for better conditioning."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model using Chebyshev polynomial basis (degrees 0-9) for well-conditioned fitting.
    
    y = sum_{k=0}^{9} p_k * T_k(2x/L - 1)
    
    where L is the maximum possible x value (6.0 from the domain [0.1, 6]).
    
    Chebyshev polynomials provide better numerical conditioning than monomials,
    which helps BFGS converge more reliably to the optimal coefficients.
    """
    x = np.asarray(x, dtype=float)
    
    # Map x from [0.1, 6] to [-1, 1] for Chebyshev evaluation
    # Using a fixed mapping based on the known domain
    L = 6.0
    z = 2.0 * x / L - 1.0
    # Clip to [-1, 1] to stay within the domain of Chebyshev polynomials
    z = np.clip(z, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T_0 through T_9 using the recurrence relation
    # T_0(z) = 1, T_1(z) = z, T_{n}(z) = 2*z*T_{n-1}(z) - T_{n-2}(z)
    T = np.zeros((10, x.shape[0]))
    T[0, :] = 1.0
    T[1, :] = z
    for n in range(2, 10):
        T[n, :] = 2.0 * z * T[n-1, :] - T[n-2, :]
    
    # Linear combination with the 10 parameters
    # params is shape (10,), T is shape (10, N)
    # We want sum_k p_k * T_k, which is params @ T
    y = np.dot(params, T)
    
    return y
# EVOLVE-BLOCK-END
