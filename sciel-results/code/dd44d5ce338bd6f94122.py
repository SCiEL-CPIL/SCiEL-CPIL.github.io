"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure x is 1D for consistent broadcasting
    x = np.asarray(x).flatten()
    
    # Define fixed RBF centers
    # 10 centers evenly spaced in [-2, 2]
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Fixed width for the Gaussian RBFs
    # sigma=0.5 provides good overlap for smooth functions over this range
    sigma = 0.5
    
    # Compute RBF activations
    # x shape: (N,)
    # centers shape: (10,)
    # We want a matrix of shape (N, 10) where M[i, j] = exp(-0.5 * ((x[i] - centers[j]) / sigma)^2)
    
    # Calculate squared distances
    # x[:, np.newaxis] has shape (N, 1)
    # centers[np.newaxis, :] has shape (1, 10)
    dist_sq = (x[:, np.newaxis] - centers[np.newaxis, :]) ** 2
    
    # Compute Gaussian kernels
    # K[i, j] = exp(-dist_sq[i, j] / (2 * sigma^2))
    K = np.exp(-dist_sq / (2.0 * sigma**2))
    
    # Linear combination of RBFs
    # params shape: (10,)
    # K shape: (N, 10)
    # prediction shape: (N,)
    prediction = K @ params
    
    return prediction
# EVOLVE-BLOCK-END
