"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    
    # Radial Basis Functions: 10 shifted, fixed-width Gaussians
    # Centers evenly spaced across [-3, 3]
    centers = np.linspace(-3.0, 3.0, 10)
    # Width: chosen so that adjacent Gaussians have reasonable overlap
    # Distance between centers is 6/9 = 2/3, use width ~ 0.8 for good overlap
    width = 0.8
    
    # Compute basis matrix: (n_points, 10)
    # Each column is exp(-((x - center_i)^2) / (2 * width^2))
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    basis = np.exp(-(diff ** 2) / (2.0 * width ** 2))
    
    return basis @ params
# EVOLVE-BLOCK-END
