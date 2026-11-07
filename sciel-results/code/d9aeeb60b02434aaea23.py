"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Fixed grid of RBF centers derived from input bounds [0.5, 2]
    # We use 2 centers per dimension to create 8 basis functions + 1 bias + 1 global linear term
    # Actually, let's use a more structured approach:
    # 4 centers in 3D space (corners of a sub-grid) + 4 more (midpoints) = 8 RBFs
    # But we only have 10 params. Let's use 6 RBFs + 4 linear/bias terms.
    
    # Centers: mix of corners and center
    # Input range is [0.5, 2.0]. Midpoint is 1.25.
    # Let's pick centers at:
    # c1: (0.5, 0.5, 0.5)
    # c2: (2.0, 2.0, 2.0)
    # c3: (1.25, 1.25, 1.25)
    # c4: (0.5, 2.0, 1.25)
    # c5: (2.0, 0.5, 1.25)
    # c6: (1.25, 0.5, 2.0)
    
    # To keep it simple and robust, let's use a fixed set of centers.
    # Using Gaussian RBFs: exp(-||p - c||^2 / (2*sigma^2))
    # We need to choose sigma. A reasonable sigma is ~0.5 or 1.0 given the range length 1.5.
    
    centers = np.array([
        [0.5, 0.5, 0.5],
        [2.0, 2.0, 2.0],
        [1.25, 1.25, 1.25],
        [0.5, 2.0, 1.25],
        [2.0, 0.5, 1.25],
        [1.25, 0.5, 2.0]
    ])
    
    sigma = 0.8
    
    # Compute RBF values for each point
    # x, y, z are arrays of shape (N,)
    points = np.stack([x, y, z], axis=1)  # (N, 3)
    
    # Compute squared distances to each center
    # dist_sq[i, j] = ||points[i] - centers[j]||^2
    # Using broadcasting
    # points[:, np.newaxis, :] shape (N, 1, 3)
    # centers[np.newaxis, :, :] shape (1, 6, 3)
    diff = points[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)  # (N, 6)
    
    # RBF values
    rbf_vals = np.exp(-dist_sq / (2 * sigma ** 2))  # (N, 6)
    
    # Linear terms: x, y, z, bias
    # We have 10 params total.
    # params[0:6] for RBFs
    # params[6:9] for linear x, y, z
    # params[9] for bias
    
    p_rbf = params[0:6]
    p_lin = params[6:9]
    p_bias = params[9]
    
    # Combine
    pred_rbf = rbf_vals @ p_rbf
    pred_lin = p_lin[0] * x + p_lin[1] * y + p_lin[2] * z
    pred_bias = p_bias
    
    return pred_rbf + pred_lin + pred_bias
# EVOLVE-BLOCK-END
