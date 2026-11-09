"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a Gaussian Radial Basis Function (RBF) network.
    params: array of length 10.
    
    We use 10 Gaussian basis functions with fixed centers and widths.
    The centers are evenly spaced across the input domain [-3, 3].
    The widths are fixed to ensure sufficient overlap for smooth interpolation.
    
    Basis functions:
    phi_i(x) = exp(-0.5 * ((x - c_i) / sigma)^2)
    
    Prediction:
    y(x) = sum_{i=0}^{9} params[i] * phi_i(x)
    
    This approach is numerically stable and avoids the Runge phenomenon associated
    with high-degree polynomials.
    """
    # Define fixed centers evenly spaced in [-3, 3]
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Define fixed width (sigma)
    # A width of ~1.2 ensures good overlap between adjacent centers (spacing ~0.67)
    # while maintaining locality.
    sigma = 1.2
    
    # Compute the basis functions
    # x shape: (n_points,)
    # centers shape: (10,)
    # We need to compute (x - c_i) for all i
    # Using broadcasting: x[:, np.newaxis] - centers[np.newaxis, :]
    x_expanded = x[:, np.newaxis]
    c_expanded = centers[np.newaxis, :]
    
    # Calculate squared distances
    diff = x_expanded - c_expanded
    
    # Calculate Gaussian values
    # phi = exp(-0.5 * (diff / sigma)^2)
    phi = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # phi shape: (n_points, 10)
    # params shape: (10,)
    
    # Prediction is the dot product
    pred = phi @ params
    
    return pred
# EVOLVE-BLOCK-END
