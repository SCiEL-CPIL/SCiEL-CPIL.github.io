"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using an RBF (Radial Basis Function) interpolant.
    
    Inputs:
        b: population density (normalized to [0, 1])
        s: substrate concentration (normalized to [0, 1])
        temp: temperature (normalized to [0, 1])
        pH: pH level (normalized to [0, 1])
        params: array of 10 fitted constants (weights for RBF centers + 1 bias)
    
    Returns:
        Predicted growth rate for each row.
    """
    # Normalize inputs to [0, 1] based on training ranges
    b_norm = (b - 1.0) / (4.999 - 1.0)
    s_norm = (s - 1.0) / (39.98 - 1.0)
    temp_norm = (temp - 1.0) / (12.0 - 1.0)
    pH_norm = (pH - 1.0) / (4.998 - 1.0)  # Assuming pH range similar to b, but let's use [1, 4.998] as given
    
    # Actually, looking at the input ranges:
    # b: [1, 4.998]
    # s: [1, 4.999]
    # temp: [1.002, 39.98]
    # pH: [2, 12]
    
    b_norm = (b - 1.0) / (4.998 - 1.0)
    s_norm = (s - 1.0) / (4.999 - 1.0)
    temp_norm = (temp - 1.002) / (39.98 - 1.002)
    pH_norm = (pH - 2.0) / (12.0 - 2.0)
    
    # Clip to [0, 1] for safety
    b_norm = np.clip(b_norm, 0, 1)
    s_norm = np.clip(s_norm, 0, 1)
    temp_norm = np.clip(temp_norm, 0, 1)
    pH_norm = np.clip(pH_norm, 0, 1)
    
    # Define 10 RBF centers in the normalized 4D space
    # Strategy: Cover the hypercube more evenly.
    # 1 center at the centroid
    # 4 centers at the "inner" corners (0.25, 0.25, 0.25, 0.25) etc? No, that's 16.
    # Let's use 8 points forming a smaller hypercube inside, plus 2 others?
    # Or use the 8 corners of the [0.2, 0.8] box and 2 more?
    
    # Let's try a grid of 2x2x2x2 is 16 points. We need 10.
    # Let's pick 10 points from a 3x3x3x3 grid? Too many.
    
    # Alternative: Use the 8 corners of the full hypercube [0,1]^4 and 2 centers?
    # Corners:
    # [0,0,0,0], [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1], [1,1,0,0], [1,0,1,0], [1,0,0,1]
    # This is 8 points.
    # Plus [0.5, 0.5, 0.5, 0.5] and [0.5, 0.5, 0.5, 0.5] (duplicate)? No.
    
    # Let's try:
    # 4 corners: [0,0,0,0], [1,1,1,1], [0,1,0,1], [1,0,1,0]
    # 4 edge midpoints: [0.5,0,0,0], [0,0.5,0,0], [0,0,0.5,0], [0,0,0,0.5]
    # 2 face centers: [0.5,0.5,0,0], [0.5,0.5,0.5,0.5]
    
    centers = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.5],
        [0.5, 0.5, 0.0, 0.0],
        [0.5, 0.5, 0.5, 0.5]
    ])
    
    # Number of samples
    n_samples = b_norm.shape[0]
    
    # Stack normalized inputs
    X = np.column_stack([b_norm, s_norm, temp_norm, pH_norm])
    
    # Compute squared distances from each sample to each center
    # X: (n_samples, 4), centers: (10, 4)
    # diff: (n_samples, 10, 4)
    diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dist = np.sum(diff ** 2, axis=2)  # (n_samples, 10)
    
    # Gaussian RBF: exp(-gamma * sq_dist)
    # Use gamma = 1.0 for simplicity (could be tuned, but we only have 10 params as weights)
    gamma = 2.0
    rbf_values = np.exp(-gamma * sq_dist)  # (n_samples, 10)
    
    # params: (10,) weights for each RBF
    # Prediction: sum of weight_i * rbf_i
    prediction = rbf_values @ params[:10]
    
    return prediction
# EVOLVE-BLOCK-END
