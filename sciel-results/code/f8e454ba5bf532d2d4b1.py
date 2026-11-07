"""Rewritten program for symbolic regression with Gaussian RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts the target function using a basis of Gaussian Radial Basis Functions (RBFs).
    
    The basis consists of 10 RBFs centered at the corners of the input domain [0.5, 2]^3,
    the center of the domain, and one additional point to break symmetry.
    The RBF width is fixed to provide local flexibility for non-polynomial structures.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if x.size == 0:
        return np.array([], dtype=np.float64)
    
    # Ensure params is 1D and has correct length
    if params.size != 10:
        # Fallback if shape is wrong, though evaluator should ensure this
        params = np.ones(10, dtype=np.float64)
        
    # Define RBF centers
    # Corners of the box [0.5, 2.0] x [0.5, 2.0] x [0.5, 2.0]
    x_min, x_max = 0.5, 2.0
    y_min, y_max = 0.5, 2.0
    z_min, z_max = 0.5, 2.0
    
    centers = np.array([
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_min, y_max, z_min],
        [x_max, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_min, y_max, z_max],
        [x_max, y_max, z_max],
        [1.25, 1.25, 1.25], # Center
        [1.25, 1.25, z_min] # Extra point
    ])
    
    # RBF width (sigma). Domain size is 1.5. 
    # Sigma ~ 0.5-0.7 is typical for this scale.
    sigma = 0.6
    sigma_sq_2 = 2.0 * sigma * sigma
    
    # Compute distances from each input point to each center
    # x, y, z are arrays of shape (N,)
    # centers is (10, 3)
    # We need to compute squared Euclidean distance for each of the N points to each of the 10 centers.
    
    # Efficient calculation:
    # Expand x, y, z to (N, 1) and subtract centers (1, 10, 3)
    # dist_sq (N, 10)
    
    x_exp = x[:, np.newaxis]
    y_exp = y[:, np.newaxis]
    z_exp = z[:, np.newaxis]
    
    dx = x_exp - centers[:, 0]
    dy = y_exp - centers[:, 1]
    dz = z_exp - centers[:, 2]
    
    dist_sq = dx**2 + dy**2 + dz**2
    
    # Compute RBF values: exp(-dist_sq / (2 * sigma^2))
    rbf_values = np.exp(-dist_sq / sigma_sq_2)
    
    # rbf_values shape: (N, 10)
    # params shape: (10,)
    
    # Prediction is linear combination
    pred = rbf_values @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
# EVOLVE-BLOCK-END
