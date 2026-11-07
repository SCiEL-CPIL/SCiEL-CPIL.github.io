"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Sparse linear combination of 10 Radial Basis Functions (RBFs).
    
    Centers are chosen from a 3x3 grid over [-1, 1]^2 plus the center (0,0).
    Widths (sigma) are varied across centers to allow both localized and broader features.
    The optimizer fits only the 10 amplitudes (params).
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    params = np.asarray(params).ravel()
    
    # 3x3 grid gives 9 points; add (0,0) for 10 total
    centers = np.array([
        [-1.0, -1.0],
        [-1.0,  0.0],
        [-1.0,  1.0],
        [ 0.0, -1.0],
        [ 0.0,  0.0],
        [ 0.0,  1.0],
        [ 1.0, -1.0],
        [ 1.0,  0.0],
        [ 1.0,  1.0],
        [ 0.0,  0.0]
    ])
    
    # Vary widths: use smaller sigma for corner/edge points (more localized)
    # and larger sigma for center points (broader influence).
    # This helps capture both sharp local features and smoother global trends.
    sigmas = np.array([
        0.5,   # (-1,-1)
        0.6,   # (-1,0)
        0.5,   # (-1,1)
        0.6,   # (0,-1)
        0.8,   # (0,0) - broadest
        0.6,   # (0,1)
        0.5,   # (1,-1)
        0.6,   # (1,0)
        0.5,   # (1,1)
        1.0    # (0,0) duplicate with even broader width for global fit
    ])
    
    # Compute squared distances
    dx = x[:, np.newaxis] - centers[:, 0][np.newaxis, :]  # (n, 10)
    dy = y[:, np.newaxis] - centers[:, 1][np.newaxis, :]  # (n, 10)
    dist_sq = dx**2 + dy**2
    
    # RBF values: exp(-dist_sq / (2 * sigma_j^2)) for each center j
    # sigmas: (10,) -> broadcast over columns
    rbf_values = np.exp(-dist_sq / (2.0 * sigmas[np.newaxis, :]**2))  # (n, 10)
    
    # Prediction
    pred = rbf_values @ params
    
    return pred
# EVOLVE-BLOCK-END
