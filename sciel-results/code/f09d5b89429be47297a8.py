"""Symbolic regression with 1 input(s) - Fourier series basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev polynomial basis of degree 9.
    
    The basis is strictly linear in params, so BFGS converges to the global
    least-squares solution. Chebyshev polynomials provide a well-conditioned
    basis for approximating smooth functions on a bounded interval, often
    outperforming Fourier series for non-periodic functions.
    
    We map the input x in [-2, 2] to t in [-1, 1] and use Chebyshev polynomials
    T_k(t) for k=0..9.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Map x from [-2, 2] to [-1, 1]
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0(t) ... T_9(t)
    # np.polynomial.chebyshev.chebvander returns a matrix where the j-th column
    # is T_j(t). We want degrees 0 to 9, so 10 columns.
    basis = np.polynomial.chebyshev.chebvander(t, 9)
    
    # Multiply basis matrix (N x 10) by params (10,)
    result = basis @ params
    
    return result
# EVOLVE-BLOCK-END
