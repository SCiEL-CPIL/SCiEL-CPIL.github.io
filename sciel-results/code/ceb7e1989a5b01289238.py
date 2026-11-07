"""Bootstrap portfolio: Chebyshev polynomial basis for better conditioning."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a Chebyshev polynomial basis.
    This basis is orthogonal on [-1, 1] and better conditioned than monomials,
    which helps the BFGS optimizer converge reliably.
    """
    # Mixed basis: Polynomials (up to degree 2), Cross term, and Non-polynomial (exp, sin)
    # This covers a wider range of potential target functions than pure polynomials.
    
    p = params
    
    # Polynomial terms
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x * x
    b4 = y * y
    b5 = x * y
    
    # Non-polynomial terms
    # exp(x) and exp(y) capture exponential growth/decay
    # sin(x) and sin(y) capture periodic behavior
    b6 = np.exp(x)
    b7 = np.exp(y)
    b8 = np.sin(x)
    b9 = np.sin(y)
    
    result = (
        p[0] * b0 +
        p[1] * b1 +
        p[2] * b2 +
        p[3] * b3 +
        p[4] * b4 +
        p[5] * b5 +
        p[6] * b6 +
        p[7] * b7 +
        p[8] * b8 +
        p[9] * b9
    )
    
    return result
# EVOLVE-BLOCK-END
