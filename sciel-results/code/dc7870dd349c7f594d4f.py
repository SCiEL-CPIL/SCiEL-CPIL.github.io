"""Symbolic regression with fixed-center Gaussian RBF network."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a fixed-center Gaussian RBF network.
    
    10 fixed centers evenly spaced in [-2, 2], with 10 learnable weights.
    The problem is convex (linear in params), so BFGS finds the global optimum.
    
    Parameters (length 10):
    w0, w1, ..., w9: weights for each Gaussian basis function
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    # Fixed centers evenly spaced in [-2, 2]
    n_centers = 10
    centers = np.linspace(-2.0, 2.0, n_centers)
    
    # Fixed width: use a moderate width that covers the domain well
    # With 10 centers in [-2,2], spacing is 4/9 ≈ 0.444
    # Use sigma such that adjacent Gaussians have reasonable overlap
    sigma = 0.5  # width parameter
    
    # Compute Gaussian basis functions: phi_i(x) = exp(-(x - c_i)^2 / (2*sigma^2))
    # Shape: (n_points, n_centers)
    x_expanded = x[:, np.newaxis]  # (n_points, 1)
    c_expanded = centers[np.newaxis, :]  # (1, n_centers)
    
    diff = x_expanded - c_expanded  # (n_points, n_centers)
    basis = np.exp(-diff ** 2 / (2.0 * sigma ** 2))  # (n_points, n_centers)
    
    # Linear combination: y = basis @ params
    y = basis @ params
    
    return y
# EVOLVE-BLOCK-END
