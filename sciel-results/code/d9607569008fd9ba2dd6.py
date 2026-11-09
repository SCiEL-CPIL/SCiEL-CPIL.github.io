"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fixed-width RBF model with 10 centers evenly spaced across [-3, 3].
    The 10 params are the RBF weights (linear-in-params).
    Centers: -3, -2.4, -1.8, -1.2, -0.6, 0, 0.6, 1.2, 1.8, 2.4
    Width: 0.5 (fixed, moderate)
    """
    # Fixed centers evenly spaced across [-3, 3]
    centers = np.array([-3.0, -2.4, -1.8, -1.2, -0.6, 0.0, 0.6, 1.2, 1.8, 2.4])
    
    # Fixed width for all RBFs
    width = 0.5
    
    # x shape: (n_points,)
    # Compute RBF features
    x_expanded = x[:, np.newaxis]  # (n_points, 1)
    centers_expanded = centers[np.newaxis, :]  # (1, 10)
    
    diff = x_expanded - centers_expanded  # (n_points, 10)
    rbf = np.exp(-diff**2 / (2.0 * width**2))  # (n_points, 10)
    
    # Weighted sum of RBFs
    return rbf @ params
# EVOLVE-BLOCK-END
