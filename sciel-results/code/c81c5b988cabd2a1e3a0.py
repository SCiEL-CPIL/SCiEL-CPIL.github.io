"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from numpy.polynomial.chebyshev import chebvander


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev polynomial approximation of degree 9 (10 parameters).
    params[0:10] -> coefficients for Chebyshev polynomials T_0..T_9 of (x/3)
    
    f(x) = sum_{i=0}^{9} c_i * T_i(x/3)
    
    Chebyshev polynomials are orthogonal on [-1, 1] and bounded by 1,
    providing a much better conditioned basis for optimization than
    standard monomials. This helps BFGS converge to the optimal coefficients.
    """
    # Scale x to [-1, 1] to map to the Chebyshev domain
    t = x / 3.0
    
    # Create Chebyshev Vandermonde matrix
    # Shape: (N, 10) where row i is [T_0(t_i), T_1(t_i), ..., T_9(t_i)]
    # chebvander returns columns as T_0, T_1, ... T_deg
    basis = chebvander(t, deg=9)
    
    # Linear combination of basis functions
    result = basis @ params
    
    return result
# EVOLVE-BLOCK-END
