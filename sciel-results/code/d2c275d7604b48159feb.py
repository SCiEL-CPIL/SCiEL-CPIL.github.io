"""
Hybrid basis: 9 RBF centers (3x3 grid) + 1 constant offset.
Combines localized Gaussian kernels (from current program) with a slightly
different width and vectorized computation for speed and numerical stability.
The RBF centers are placed on a 3x3 grid covering [-1,1]^2, and the width is
chosen to provide good local coverage while maintaining smoothness.
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure inputs are 1D arrays
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    
    # Fixed 3x3 grid of centers in [-1, 1]^2
    cx = np.array([-1.0, 0.0, 1.0])
    cy = np.array([-1.0, 0.0, 1.0])
    
    # Gaussian kernel width parameter (gamma = 1/(2*sigma^2))
    # Using gamma=4.0 for a balance between locality and smoothness
    gamma = 4.0
    
    # Pre-compute RBF activations vectorized
    # x, y have shape (N,)
    # cx, cy have shape (3,)
    
    # Compute squared distances to each of the 9 centers
    # We'll build a (N, 9) array of RBF values
    
    # Centers as (9, 2) array
    centers_x = np.array([cx[i] for i in range(3) for j in range(3)])
    centers_y = np.array([cy[j] for i in range(3) for j in range(3)])
    
    # dx, dy have shape (N, 9)
    dx = x[:, None] - centers_x[None, :]
    dy = y[:, None] - centers_y[None, :]
    dist_sq = dx * dx + dy * dy
    
    # RBF activations: exp(-gamma * dist_sq), shape (N, 9)
    rbf_vals = np.exp(-gamma * dist_sq)
    
    # params[0:9] are weights for the 9 RBFs, params[9] is the offset
    # Linear combination
    result = rbf_vals @ params[:9] + params[9]
    
    return result
# EVOLVE-BLOCK-END
