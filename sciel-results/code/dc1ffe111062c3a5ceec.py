"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Gaussian Radial Basis Function (RBF) network.
    The first five parameters define the centers of the kernels, and the last five
    define their amplitudes. The widths are fixed to allow BFGS to optimize the
    non-convex center locations and amplitudes effectively.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Extract centers and amplitudes
    centers = params[0:5]
    amplitudes = params[5:10]
    
    # Fixed width for the Gaussian kernels
    # A width of 1.0 covers a significant portion of the [-2, 2] domain
    sigma = 1.0
    sigma2 = 2.0 * sigma * sigma
    
    result = np.zeros_like(x)
    
    # Sum of Gaussian kernels
    for i in range(5):
        c = centers[i]
        a = amplitudes[i]
        # Calculate Gaussian kernel values
        diff = x - c
        kernel = np.exp(-0.5 * (diff * diff) / sigma2)
        result += a * kernel
        
    return result
# EVOLVE-BLOCK-END
