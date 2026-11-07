"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # New Design: Mixture of 5 Gaussians with adjustable centers and signed weights.
    # This allows the model to shift the "bumps" to fit the data and represent negative values.
    # params[0:5] : offsets added to base centers
    # params[5:10] : weights for the kernels (can be negative)
    
    # Base centers evenly spaced in [0.1, 6.0]
    base_centers = np.linspace(0.1, 6.0, 5)
    
    # Apply offsets to centers
    # Initial params are 1, so centers will be shifted by +1 initially.
    # This is acceptable as the optimizer will adjust them.
    centers = base_centers + params[0:5]
    
    # Weights can be positive or negative
    weights = params[5:10]
    
    # Fixed bandwidth
    # With 5 centers, spacing is ~1.47. Sigma=1.0 provides good overlap for smooth functions.
    sigma = 1.0
    
    # x shape: (n,)
    # centers shape: (5,)
    # Compute distances: (n, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Squared exponential kernel: exp(-diff^2 / (2 * sigma^2))
    kernel = np.exp(-diff ** 2 / (2.0 * sigma ** 2))
    
    # Prediction: sum of weighted kernels
    # (n, 5) @ (5,) -> (n,)
    pred = kernel @ weights
    
    return pred
# EVOLVE-BLOCK-END
