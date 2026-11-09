"""Symbolic regression with 1 input(s) using a pure Multi-Scale RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the unknown function using a linear combination of 10 Gaussian Radial Basis Functions (RBFs).
    
    Structural Change:
    - Abandons the hybrid Chebyshev-Fourier-Exponential basis which may suffer from multicollinearity 
      and ill-conditioning, hindering BFGS convergence.
    - Adopts a Pure RBF basis, which is generally better conditioned for local and global smooth functions.
    - Uses 10 RBFs to fully utilize the 10 parameters for local flexibility.
    - Centers are evenly spaced across the domain [0.1, 6.0].
    - Bandwidth (sigma) is set to 1.2 to ensure sufficient overlap between adjacent RBFs, 
      providing a smooth global structure while retaining local adaptability. This wider bandwidth 
      compared to the previous program (sigma=0.8) improves the condition number of the basis matrix.
    """
    x = np.asarray(x, dtype=np.float64)
    
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
        
    n_points = x.shape[0]
    if n_points == 0:
        return np.array([])
        
    # Define 10 RBF centers spread across the domain [0.1, 6.0]
    centers = np.linspace(0.1, 6.0, 10)
    
    # Width parameter for RBFs (Gaussian)
    # sigma = 1.2 provides good overlap and smoothness
    sigma = 1.2
    
    # Build the RBF basis matrix
    # Shape: (n_points, 10)
    # basis[:, i] = exp(-(x - centers[i])^2 / (2 * sigma^2))
    
    # Vectorized computation for speed
    # diff is (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    basis = np.exp(-(diff ** 2) / (2.0 * sigma ** 2))
    
    # Linear combination
    # params is shape (10,)
    pred = basis @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        return np.zeros_like(x)
        
    return pred
# EVOLVE-BLOCK-END
