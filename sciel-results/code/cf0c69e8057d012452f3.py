"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # 5 fixed evenly-spaced centers across [-3, 3]
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    
    # Extract 5 weights and 5 widths from params
    weights = params[:5]
    widths = params[5:10]
    
    # Ensure widths are positive (use softplus-like transformation)
    # widths_eff = np.maximum(widths, 0.1) to avoid division by zero
    widths_eff = np.abs(widths) + 0.1
    
    # Compute RBF features: exp(-((x - center)^2) / (2 * width^2))
    x_expanded = x[:, np.newaxis]
    centers_expanded = centers[np.newaxis, :]
    
    # Shape: (n_points, 5)
    diff = x_expanded - centers_expanded
    rbf = np.exp(-diff**2 / (2 * widths_eff**2))
    
    # Weighted sum of RBFs
    return rbf @ weights
# EVOLVE-BLOCK-END
