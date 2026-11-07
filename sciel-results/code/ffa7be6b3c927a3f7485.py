"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Use Legendre polynomials as a basis for symbolic regression.
    Legendre polynomials are orthogonal on [-1, 1], which gives a well-conditioned
    design matrix and a smoother optimization landscape for BFGS.
    
    We map x from [-2, 2] to [-1, 1] and use the first 10 Legendre polynomials.
    The 10 params are the coefficients of these polynomials.
    """
    # Map x from [-2, 2] to [-1, 1]
    t = (x - 0.0) / 2.0  # x is already in [-2, 2], so t = x/2 is in [-1, 1]
    
    # Compute Legendre polynomials P_0 through P_9 using recurrence relation
    # P_0(t) = 1
    # P_1(t) = t
    # P_n(t) = ((2n-1)*t*P_{n-1}(t) - (n-1)*P_{n-2}(t)) / n
    
    # Build the design matrix: shape (N, 10)
    N = len(x)
    P = np.zeros((N, 10))
    P[:, 0] = 1.0
    if N > 0:
        P[:, 1] = t
        for n in range(2, 10):
            P[:, n] = ((2 * n - 1) * t * P[:, n - 1] - (n - 1) * P[:, n - 2]) / n
    
    # Return linear combination of Legendre polynomials
    return P @ params
# EVOLVE-BLOCK-END
