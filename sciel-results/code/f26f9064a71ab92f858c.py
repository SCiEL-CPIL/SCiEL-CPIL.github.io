"""RBF-based symbolic regression with fixed centers and widths."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of 10 Gaussian RBFs with fixed centers and widths.
    Centers are evenly spaced across [0.1, 6].
    Widths are chosen so that adjacent RBFs have moderate overlap.
    """
    x_min = 0.1
    x_max = 6.0
    
    # 10 evenly spaced centers
    centers = np.linspace(x_min, x_max, 10)
    
    # Fixed width: distance between adjacent centers is (x_max - x_min) / 9
    # Use width = 0.5 * spacing for moderate overlap
    spacing = (x_max - x_min) / 9.0
    width = 0.5 * spacing
    
    # Compute RBF values: phi_i(x) = exp(-((x - c_i)^2) / (2 * width^2))
    # x shape: (N,), centers shape: (10,)
    # We want output shape (N, 10) where column i is phi_i(x)
    
    # Efficient computation using broadcasting
    x_col = x[:, np.newaxis]  # (N, 1)
    centers_row = centers[np.newaxis, :]  # (1, 10)
    
    # Squared distances
    sq_dist = (x_col - centers_row) ** 2  # (N, 10)
    
    # Gaussian RBFs
    rbf_values = np.exp(-sq_dist / (2.0 * width ** 2))  # (N, 10)
    
    # Linear combination with params
    return rbf_values @ params
# EVOLVE-BLOCK-END
