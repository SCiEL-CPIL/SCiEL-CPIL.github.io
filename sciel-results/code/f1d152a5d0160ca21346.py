"""E. coli growth rate model: RBF kernel regression on normalized inputs."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a linear-in-parameters RBF kernel expansion.
    
    The prediction is a weighted sum of Gaussian similarities between the normalized
    4D input vector and a fixed set of grid points. This allows the model to learn
    a smooth, high-dimensional response surface. The 10 parameters are the weights
    for the 10 grid points (or features), making the model linear in params.
    
    Inputs:
    b: population density [1, 5]
    s: substrate [1, 5]
    temp: temperature [1, 40]
    pH: [2, 12]
    
    Parameters (10):
    p0-p9: Weights for the 10 RBF basis functions centered at grid points.
    """
    # Normalize inputs to [0, 1]
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    temp_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    # Define a fixed set of 10 grid points in the 4D normalized space
    # We choose points that cover the range and capture typical E. coli optima
    # E. coli optimal temp ~37C -> (37-1)/39 ~ 0.92
    # E. coli optimal pH ~7 -> (7-2)/10 = 0.5
    # Substrate saturation: s_norm around 0.5-1.0
    # Density: b_norm around 0.0-0.5
    
    # Grid points (b, s, temp, ph) in normalized space
    centers = np.array([
        [0.25, 0.25, 0.5, 0.5],   # Low temp, low substrate
        [0.25, 0.75, 0.5, 0.5],   # Low temp, high substrate
        [0.75, 0.25, 0.5, 0.5],   # High temp, low substrate
        [0.75, 0.75, 0.5, 0.5],   # High temp, high substrate
        [0.25, 0.5, 0.92, 0.5],   # Optimal temp, low density
        [0.75, 0.5, 0.92, 0.5],   # Optimal temp, high density
        [0.25, 0.5, 0.5, 0.3],    # Low pH
        [0.75, 0.5, 0.5, 0.7],    # High pH
        [0.5, 0.5, 0.92, 0.5],    # Center density, optimal temp
        [0.5, 0.9, 0.92, 0.5],    # High substrate, optimal temp
    ])
    
    # Input vector shape: (N, 4)
    inputs = np.stack([b_norm, s_norm, temp_norm, ph_norm], axis=-1)
    
    # Compute squared distances from each input to each center
    # inputs: (N, 1, 4), centers: (1, 10, 4)
    diff = inputs[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dist = np.sum(diff ** 2, axis=-1)  # (N, 10)
    
    # RBF kernel: exp(-gamma * sq_dist)
    # Use a fixed gamma to control the width of the RBFs
    # gamma = 1/(2 * sigma^2). Let's use sigma=0.3 in normalized space.
    gamma = 1.0 / (2.0 * 0.3 ** 2)
    
    rbf_values = np.exp(-gamma * sq_dist)  # (N, 10)
    
    # Linear combination of RBF values with parameters
    # params: (10,)
    result = rbf_values @ params
    
    # Ensure finite output
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
