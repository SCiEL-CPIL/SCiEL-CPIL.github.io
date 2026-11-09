"""Symbolic regression with linear Chebyshev polynomial basis (10 params)."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: y(x) = sum_{k=0}^{9} c_k * T_k(t)
    
    Where t is x mapped from [-3, 3] to [-1, 1].
    T_k(t) are Chebyshev polynomials of the first kind.
    
    This is a linear model in the parameters c_k.
    Using Chebyshev polynomials provides a stable, orthogonal basis 
    for function approximation on the interval [-1, 1], which helps
    avoid ill-conditioning compared to standard monomial bases (x, x^2, ...).
    
    Parameters:
      c_0 ... c_9 : Coefficients for T_0 ... T_9
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Map x from [-3, 3] to [-1, 1]
    # t = (x - 0) / 3
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_0(t) to T_9(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    
    basis = np.zeros((10, x.size), dtype=np.float64)
    basis[0, :] = 1.0
    if x.size > 0:
        basis[1, :] = t
        
        for k in range(2, 10):
            basis[k, :] = 2.0 * t * basis[k-1, :] - basis[k-2, :]
            
    # Linear combination
    # y = basis.T @ params
    y = np.dot(basis.T, params)
    
    return y
# EVOLVE-BLOCK-END
