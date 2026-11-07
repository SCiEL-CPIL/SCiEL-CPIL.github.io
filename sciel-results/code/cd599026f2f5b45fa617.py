import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using a Radial Basis Function (RBF) model.
    
    The inputs are normalized to [0, 1] based on the known training ranges.
    10 Gaussian basis functions are defined with fixed centers and a fixed width.
    The 10 parameters are the linear weights for these basis functions.
    This transforms the problem into a linear regression on non-linear features,
    which is convex with respect to the parameters and easily solvable by BFGS.
    """
    
    # Known ranges from the task description
    b_min, b_max = 1.0, 4.998
    s_min, s_max = 1.0, 4.999
    temp_min, temp_max = 1.002, 39.98
    ph_min, ph_max = 2.0, 12.0
    
    # Normalize inputs to [0, 1] to make the RBF centers and widths scale-invariant
    # Add small epsilon to avoid division by zero if ranges were identical (they are not)
    b_norm = (b - b_min) / (b_max - b_min)
    s_norm = (s - s_min) / (s_max - s_min)
    temp_norm = (temp - temp_min) / (temp_max - temp_min)
    ph_norm = (pH - ph_min) / (ph_max - ph_min)
    
    # Stack normalized inputs into a matrix of shape (N, 4)
    # N is the number of rows
    X = np.column_stack((b_norm, s_norm, temp_norm, ph_norm))
    
    # Define 10 fixed centers in the normalized [0, 1]^4 space
    # We choose centers to cover the corners, some edges, and the center
    centers = np.array([
        [0.0, 0.0, 0.0, 0.0],  # Corner 1
        [1.0, 1.0, 1.0, 1.0],  # Corner 2
        [0.0, 1.0, 0.0, 1.0],  # Corner 3
        [1.0, 0.0, 1.0, 0.0],  # Corner 4
        [0.0, 0.0, 1.0, 1.0],  # Corner 5
        [1.0, 1.0, 0.0, 0.0],  # Corner 6
        [0.0, 1.0, 1.0, 0.0],  # Corner 7
        [1.0, 0.0, 0.0, 1.0],  # Corner 8
        [0.5, 0.5, 0.5, 0.5],  # Center
        [0.25, 0.75, 0.25, 0.75] # Additional point for coverage
    ])
    
    # Fixed width for the Gaussian basis functions
    # sigma = 0.4 provides a good balance between local and global influence in [0,1] space
    sigma = 0.4
    
    # Compute the squared distances from each point X to each center C
    # X: (N, 4), Centers: (10, 4)
    # We want dist_sq: (N, 10)
    # dist_sq[i, j] = ||X[i] - Centers[j]||^2
    # Using broadcasting: (N, 1, 4) - (1, 10, 4) -> (N, 10, 4)
    diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    
    # Compute the RBF values: exp(-dist_sq / (2 * sigma^2))
    # Shape: (N, 10)
    rbf_values = np.exp(-dist_sq / (2.0 * sigma ** 2))
    
    # The parameters are the weights for these 10 basis functions
    # params: (10,)
    # Prediction: (N, 10) @ (10,) -> (N,)
    prediction = rbf_values @ params
    
    return prediction
