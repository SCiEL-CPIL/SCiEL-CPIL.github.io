"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Separable RBF representation with 10 fixed 2D Gaussian kernels.
    Centers at vertices and midpoints of the unit square [-1,1]^2.
    params: 10 weights for each kernel.
    
    The output is: sum_i params[i] * exp(-((x-cx_i)^2 + (y-cy_i)^2) / (2*sigma^2))
    
    This is linear in params, so BFGS will find the global minimum.
    The fixed sigma is chosen to be ~0.5 to balance locality and coverage.
    """
    # Fixed centers: 4 corners + 4 midpoints of edges + 2 additional for coverage
    # Actually, let's use a more systematic set of 10 centers
    # Corners: (-1,-1), (-1,1), (1,-1), (1,1) -> 4
    # Edge midpoints: (-1,0), (1,0), (0,-1), (0,1) -> 4
    # Center and one more: (0,0), and maybe (0.5, 0.5) or just use a 3x3 grid minus one
    # Let's use: 4 corners, 4 edge midpoints, center (0,0), and one more at (0.5, 0.5)
    # Actually, for better coverage, let's use a 3x3 grid:
    # (-1,-1), (-1,0), (-1,1)
    # (0,-1),  (0,0),  (0,1)
    # (1,-1),  (1,0),  (1,1)
    # That's 9 points. We need 10. Let's add (0.5, 0.5) or use a different arrangement.
    # Let's stick with the 3x3 grid (9 points) and add one more at (0.5, -0.5)
    # Or better: use the 10 points from the directive: vertices and midpoints
    # Vertices: (-1,-1), (-1,1), (1,-1), (1,1)
    # Midpoints of edges: (-1,0), (1,0), (0,-1), (0,1)
    # That's 8. Add center (0,0) and one more... let's add (0.5, 0.5)
    
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
        [ 0.5,  0.5]
    ])
    
    # Fixed bandwidth
    sigma = 0.6
    
    # Compute distances from each point to each center
    # x, y are arrays of shape (n,)
    # We want a matrix of shape (n, 10)
    dx = x[:, np.newaxis] - centers[0, :][np.newaxis, :]  # (n, 1)
    dy = y[:, np.newaxis] - centers[1, :][np.newaxis, :]  # (n, 1)
    
    # Actually, let's compute more efficiently
    # dist_sq[i, j] = (x[i] - cx[j])^2 + (y[i] - cy[j])^2
    cx = centers[:, 0]  # (10,)
    cy = centers[:, 1]  # (10,)
    
    # (n, 10)
    dx = x[:, np.newaxis] - cx[np.newaxis, :]
    dy = y[:, np.newaxis] - cy[np.newaxis, :]
    dist_sq = dx * dx + dy * dy
    
    # Compute RBF values: exp(-dist_sq / (2 * sigma^2))
    rbf_vals = np.exp(-dist_sq / (2.0 * sigma * sigma))  # (n, 10)
    
    # Prediction: rbf_vals @ params
    pred = rbf_vals @ params
    
    return pred
# EVOLVE-BLOCK-END
