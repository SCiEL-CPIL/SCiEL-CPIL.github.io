"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    # 10 localized Gaussian basis functions evenly spaced across [-3, 3]
    # centers: -3, -2.4, -1.8, -1.2, -0.6, 0, 0.6, 1.2, 1.8, 2.4
    # (10 points, spacing 0.6)
    centers = np.linspace(-3.0, 2.4, 10)
    # Fixed widths: moderate width to ensure smooth overlap but local flexibility
    # Using sigma = 0.8 gives good overlap for smooth functions
    sigma = 0.8
    # Precompute the basis matrix: shape (n_points, 10)
    # basis[i, j] = exp(-0.5 * ((x_i - centers_j) / sigma)^2)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    basis = np.exp(-0.5 * (diff / sigma) ** 2)
    # Prediction is linear combination: basis @ params
    pred = basis @ params
    return pred
# EVOLVE-BLOCK-END
