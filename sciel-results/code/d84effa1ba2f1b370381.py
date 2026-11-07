"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev polynomial basis on [-2, 2] mapped to [-1, 1].
    
    Chebyshev polynomials provide a well-conditioned, nearly orthogonal basis
    that allows BFGS to efficiently find the global least-squares optimum.
    This can approximate any continuous function on the domain with 10 coefficients.
    
    We use 10 Chebyshev polynomials T_0 through T_9.
    """
    t = x / 2.0  # Map from [-2, 2] to [-1, 1]
    
    # Compute Chebyshev polynomials T_k(t) for k=0..9
    # T_0(t) = 1
    # T_1(t) = t
    # T_{k+1}(t) = 2*t*T_k(t) - T_{k-1}(t)
    
    T0 = np.ones_like(t)
    T1 = t
    
    # Build array of Chebyshev polynomials
    cheb = [T0, T1]
    for k in range(2, 10):
        Tk = 2.0 * t * cheb[-1] - cheb[-2]
        cheb.append(Tk)
    
    # Stack basis functions: shape (n_points, 10)
    basis = np.stack(cheb, axis=-1)
    
    # Linear combination
    result = basis @ params
    
    return result
# EVOLVE-BLOCK-END
