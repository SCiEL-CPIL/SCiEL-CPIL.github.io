"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # params is length 10
    # We construct a basis of 10 functions:
    # f0 = 1
    # f1 = x
    # f2 = x^2
    # f3 = x^3
    # f4 = x^4
    # f5 = x^5
    # f6 = x^6
    # f7 = x^7
    # f8 = x^8
    # f9 = x^9
    # This is just a monomial basis up to degree 9.
    # However, monomial bases can be ill-conditioned.
    # Let's use a scaled monomial basis or Chebyshev.
    # The task says inputs are in [-2, 2].
    # Let's map to [-1, 1] to use standard Chebyshev polynomials T_k(t).
    # T_0(t) = 1
    # T_1(t) = t
    # T_{k}(t) = 2t T_{k-1}(t) - T_{k-2}(t)
    
    # But wait, BFGS is used to fit the params.
    # If the basis is linear, BFGS should find the global minimum exactly (since it's a quadratic problem).
    # The main risk is numerical instability if the basis functions are highly correlated or ill-conditioned.
    # Chebyshev polynomials on [-1, 1] are well-conditioned.
    
    t = x / 2.0  # Map [-2, 2] to [-1, 1]
    
    # Compute Chebyshev polynomials T_0(t) through T_9(t)
    # T_0 = 1
    # T_1 = t
    # T_2 = 2t^2 - 1
    # ...
    
    # We can compute these recursively.
    # T_k(t)
    
    # Let's create an array of shape (10, N) where N is len(x)
    N = len(x)
    basis = np.zeros((10, N))
    
    basis[0, :] = 1.0
    if N > 0:
        basis[1, :] = t
        
        for k in range(2, 10):
            basis[k, :] = 2 * t * basis[k-1, :] - basis[k-2, :]
    
    # The model is sum_{i=0}^{9} params[i] * basis[i, :]
    # This is equivalent to basis.T @ params
    # We need to return an array of shape (N,)
    
    y = np.dot(basis.T, params)
    
    return y
# EVOLVE-BLOCK-END
