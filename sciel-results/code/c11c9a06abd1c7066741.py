"""Rewritten program using a hybrid polynomial and Gaussian RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Precompute fixed basis components
# Polynomial part: 1, t, t^2, t^3, t^4 where t = x/2
# RBF part: 5 Gaussians centered at t = -1, -0.5, 0, 0.5, 1 with sigma_t = 0.5

# Centers in t-space
centers_t = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
sigma_t = 0.5

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model:
    y = sum_{k=0}^4 p_k (x/2)^k + sum_{j=0}^4 p_{5+j} exp(-(x/2 - c_j)^2 / (2 * sigma_t^2))
    
    params[0:5] are polynomial coefficients.
    params[5:10] are RBF amplitudes.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return np.array([], dtype=float)
    
    # Scale x to [-1, 1] for better conditioning
    t = x / 2.0
    
    # Polynomial basis (5 columns)
    # P_0=1, P_1=t, P_2=t^2, P_3=t^3, P_4=t^4
    poly_basis = np.ones((n, 5))
    poly_basis[:, 1] = t
    poly_basis[:, 2] = t**2
    poly_basis[:, 3] = t**3
    poly_basis[:, 4] = t**4
    
    # RBF basis (5 columns)
    # diff = t[:, None] - centers_t[None, :]
    # rbf = exp(-diff^2 / (2 * sigma_t^2))
    diff = t[:, None] - centers_t[None, :]
    rbf_basis = np.exp(-(diff**2) / (2.0 * sigma_t**2))
    
    # Combine basis
    # Full basis is (n, 10)
    # First 5 cols are poly, last 5 are rbf
    # We can just concatenate or do two dot products
    y_poly = poly_basis @ params[:5]
    y_rbf = rbf_basis @ params[5:]
    
    return y_poly + y_rbf
# EVOLVE-BLOCK-END
