"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a linear combination of 10 Radial Basis Functions (RBFs).
    
    The 10 parameters are used as follows:
    - params[0]: shared width parameter (sigma) for all RBFs.
    - params[1:11]: weights for the 10 RBF centers.
    
    The centers are fixed and evenly spaced across the domain [-3, 3].
    This approach provides a smooth, well-conditioned approximation that BFGS can optimize efficiently.
    """
    # Extract parameters
    sigma = params[0]
    weights = params[1:11]
    
    # Ensure sigma is positive and not too small to avoid numerical issues
    # We use a softplus-like transformation to keep sigma positive, but since BFGS
    # starts at 1.0 and we want smoothness, we can just clamp or use abs.
    # However, the task says "single learnable width parameter", and BFGS optimizes params directly.
    # If sigma becomes negative, the RBF formula exp(-0.5 * ((x-c)/sigma)^2) would still work
    # because sigma is squared. But if sigma is 0, we get division by zero.
    # Let's ensure sigma is at least a small positive number.
    sigma_safe = np.maximum(np.abs(sigma), 1e-6)
    
    # Define fixed, evenly spaced centers in [-3, 3]
    # 10 centers
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Compute RBF values
    # x is an array of shape (N,)
    # centers is an array of shape (10,)
    # We want to compute for each x_i, the sum over j of weights[j] * exp(-0.5 * ((x_i - centers[j]) / sigma_safe)^2)
    
    # Reshape for broadcasting: x[:, np.newaxis] has shape (N, 1), centers[np.newaxis, :] has shape (1, 10)
    x_expanded = x[:, np.newaxis]
    centers_expanded = centers[np.newaxis, :]
    
    # Compute squared distances
    diff = x_expanded - centers_expanded
    squared_dist = diff ** 2
    
    # Compute RBF values: exp(-0.5 * (dist/sigma)^2)
    rbf_values = np.exp(-0.5 * squared_dist / (sigma_safe ** 2))
    
    # Linear combination: sum over j of weights[j] * rbf_values[:, j]
    # rbf_values has shape (N, 10), weights has shape (10,)
    prediction = rbf_values @ weights
    
    return prediction
# EVOLVE-BLOCK-END
