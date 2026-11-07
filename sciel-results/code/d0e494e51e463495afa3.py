"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a fixed Gaussian basis.
    This allows the model to fit functions with local variations or bumps.
    
    params: array of length 10, representing amplitudes for 10 Gaussian kernels.
    """
    x = np.asarray(x)
    
    # Define 10 fixed centers in [-3, 3]
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Fixed standard deviation for the Gaussians
    sigma = 1.0
    
    # Compute the Gaussian basis values
    # x shape: (N,)
    # centers shape: (10,)
    # diff shape: (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian kernel: exp(-0.5 * (diff/sigma)^2)
    basis = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Linear combination of basis functions
    # basis shape: (N, 10)
    # params shape: (10,)
    pred = basis @ params
    
    return pred
# EVOLVE-BLOCK-END
