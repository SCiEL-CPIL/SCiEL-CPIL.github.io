"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid Chebyshev + RBF basis for flexible fitting.
    Uses 4 Chebyshev polynomials (T0-T3) and 6 RBFs with fixed centers.
    y = sum_{i=0}^{3} p_i * T_i(x_norm) + sum_{j=4}^{9} p_j * exp(-sigma^2 * (x - c_j)^2)
    where x_norm = (x - 3.05) / 2.95 maps [0.1, 6] to [-1, 1].
    """
    x = np.asarray(x, dtype=float)
    
    # Normalize x to [-1, 1] for Chebyshev stability
    x_norm = (x - 3.05) / 2.95
    
    # Chebyshev polynomials: T0=1, T1=x, T2=2x^2-1, T3=4x^3-3x
    T0 = np.ones_like(x_norm)
    T1 = x_norm
    T2 = 2.0 * x_norm**2 - 1.0
    T3 = 4.0 * x_norm**3 - 3.0 * x_norm
    
    # RBF centers spread across [0.1, 6]
    centers = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    sigma = 0.8
    
    # Compute RBF terms
    rbf_terms = np.exp(-sigma**2 * (x[:, None] - centers[None, :])**2)
    
    # Combine: first 4 params for Chebyshev, last 6 for RBF
    cheb_part = params[0] * T0 + params[1] * T1 + params[2] * T2 + params[3] * T3
    rbf_part = np.sum(rbf_terms * params[4:10][None, :], axis=1)
    
    return cheb_part + rbf_part
# EVOLVE-BLOCK-END
