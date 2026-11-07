"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Kernel Ridge Regression strategy with Gaussian RBF kernels.
    # Use 9 evenly spaced RBF centers in [-3, 3] plus a constant feature (1.0)
    # so the 10 parameters can fit both local shape and global offset.
    
    # 9 centers evenly spaced in [-3, 3]
    centers = np.linspace(-3.0, 3.0, 9)
    
    # Bandwidth: sigma = 0.9 -> gamma = 1/(2*sigma^2) ≈ 0.6173
    gamma = 1.0 / (2 * 0.9 ** 2)
    
    # Compute distances from each input point to each center
    # x shape: (n,), centers shape: (9,)
    # diff shape: (n, 9)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Compute Gaussian RBF values
    # rbf shape: (n, 9)
    rbf = np.exp(-gamma * diff ** 2)
    
    # Append a constant feature for the global offset
    # constant shape: (n, 1)
    constant = np.ones((x.shape[0], 1), dtype=rbf.dtype)
    
    # Combined feature matrix shape: (n, 10)
    features = np.hstack((rbf, constant))
    
    # Linear combination with params (10,)
    # prediction shape: (n,)
    prediction = features @ params
    
    return prediction
# EVOLVE-BLOCK-END
