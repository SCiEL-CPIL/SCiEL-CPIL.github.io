"""Symbolic regression with fixed-center RBF basis for stable convex fitting."""
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter Gaussian RBF network with fixed centers and widths.
    
    Centers are evenly spaced in [-2, 2].
    Widths are chosen to provide good coverage.
    Only the linear combination weights (params) are optimized.
    This makes the optimization problem convex (linear least squares in params),
    avoiding local minima that plague nonlinear parameter optimization.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Fixed centers: 10 evenly spaced points in [-2, 2]
    n_centers = 10
    centers = np.linspace(-2.0, 2.0, n_centers)
    
    # Fixed widths: chosen to give each RBF reasonable coverage
    # The spacing between centers is 4/(n_centers-1) = 4/9 ≈ 0.444
    # A width of about 0.5-0.8 gives good overlap without being too broad
    width = 0.7
    
    # Compute RBF basis matrix: shape (len(x), n_centers)
    # phi[i, j] = exp(-((x_i - center_j)^2) / (2 * width^2))
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    basis = np.exp(-(diff ** 2) / (2.0 * width ** 2))
    
    # Prediction is linear combination of basis functions
    return basis @ params
