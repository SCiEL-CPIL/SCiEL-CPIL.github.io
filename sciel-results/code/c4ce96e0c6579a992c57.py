"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    GP-inspired non-parametric model using 5 Gaussian basis functions with
    shared and per-center length-scales. The 10 parameters define:
    - 5 weights (one per center)
    - 4 length-scale parameters (mapped via exp for positivity)
    - 1 global variance/offset parameter
    
    This provides local flexibility to capture non-linear structure without
    being constrained to a global polynomial basis.
    """
    # Fixed centers chosen to spread across [0.5, 2]^3
    centers = np.array([
        [1.0, 1.0, 1.0],       # center
        [0.7, 0.7, 0.7],       # low corner
        [1.8, 1.8, 1.8],       # high corner
        [0.7, 1.8, 1.0],       # mixed
        [1.8, 0.7, 1.0],       # mixed
    ])
    
    # Extract parameters
    weights = params[0:5]
    
    # Map length-scale parameters to positive values via exp
    # Use a soft clipping to avoid extreme values
    log_l1 = params[5]
    log_l2 = params[6]
    log_l3 = params[7]
    log_l4 = params[8]
    
    l1 = np.exp(np.clip(log_l1, -5, 2))
    l2 = np.exp(np.clip(log_l2, -5, 2))
    l3 = np.exp(np.clip(log_l3, -5, 2))
    l4 = np.exp(np.clip(log_l4, -5, 2))
    
    # Global scale parameter
    global_scale = params[9]
    
    # Compute distances from each data point to each center
    # x, y, z are 1D arrays of shape (N,)
    X = np.column_stack((x, y, z))  # (N, 3)
    C = centers  # (5, 3)
    
    # Squared distances: (N, 5)
    diff = X[:, np.newaxis, :] - C[np.newaxis, :, :]  # (N, 5, 3)
    dist_sq = np.sum(diff ** 2, axis=2)  # (N, 5)
    
    # Per-center length-scales: use combinations of l1-l4
    # Center 0: l1, Center 1: l2, Center 2: l3, Center 3: l4, Center 4: l1
    length_scales = np.array([l1, l2, l3, l4, l1])  # (5,)
    
    # Gaussian basis values: exp(-dist_sq / (2 * l^2))
    # Shape: (N, 5)
    rbf_values = np.exp(-dist_sq / (2.0 * length_scales[np.newaxis, :] ** 2))
    
    # Weighted sum
    predictions = rbf_values @ weights
    
    # Add global offset
    predictions = predictions + global_scale
    
    return predictions
# EVOLVE-BLOCK-END
