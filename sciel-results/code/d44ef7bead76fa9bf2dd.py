"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: Chebyshev polynomials (well-conditioned, smooth approximation)
    combined with a few targeted non-polynomial features (Gaussian bump, tanh)
    to capture potential localized or saturating behaviors in the target function.
    
    The model remains linear in parameters, ensuring a convex optimization landscape
    for BFGS.
    """
    # Scale input to [-1, 1] for Chebyshev polynomials
    u = x / 3.0
    
    n_points = len(x)
    if n_points == 0:
        return np.array([])
    
    # Construct basis matrix B of shape (n_points, 10)
    B = np.zeros((n_points, 10))
    
    # First 7 columns: Chebyshev polynomials T_0 to T_6
    # T_0(u) = 1
    # T_1(u) = u
    # T_k(u) = 2*u*T_{k-1}(u) - T_{k-2}(u)
    B[:, 0] = 1.0
    B[:, 1] = u
    
    for k in range(2, 7):
        B[:, k] = 2.0 * u * B[:, k-1] - B[:, k-2]
    
    # Column 7: Gaussian bump exp(-u^2)
    # This helps if the target has a peak or localized feature near 0
    B[:, 7] = np.exp(-(u * u))
    
    # Column 8: tanh(u)
    # This helps if the target saturates or has a step-like transition
    B[:, 8] = np.tanh(u)
    
    # Column 9: Higher order Chebyshev T_7
    # Extends the polynomial capability slightly beyond degree 6
    # T_7 = 2*u*T_6 - T_5
    B[:, 9] = 2.0 * u * B[:, 6] - B[:, 5]
    
    # Linear combination
    return B @ params
# EVOLVE-BLOCK-END
