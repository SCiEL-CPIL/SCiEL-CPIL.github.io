# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a combination of Radial Basis Functions (RBFs),
    a linear term, and a bias. This model is non-linear in the parameters (widths),
    allowing it to capture a wider variety of smooth functions than a linear basis.
    """
    x = np.asarray(x, dtype=float).flatten()
    
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
        
    n_points = len(x)
    if n_points == 0:
        return np.array([])

    # Extract parameters
    # p[0:4] are weights for 4 RBFs
    # p[4:8] are width parameters (squared) for 4 RBFs
    # p[8] is linear coefficient
    # p[9] is bias
    w = params[0:4]
    widths_sq = params[4:8]
    lin_coeff = params[8]
    bias = params[9]
    
    # Fixed centers for the RBFs, spaced across [-3, 3]
    centers = np.array([-1.5, -0.5, 0.5, 1.5])
    
    # Ensure widths are positive and non-zero to avoid division by zero or instability
    # We use (widths_sq^2 + epsilon) to ensure the coefficient in the exponent is positive
    # and differentiable.
    eps = 1e-6
    effective_widths = widths_sq**2 + eps
    
    # Initialize prediction with linear and bias terms
    pred = lin_coeff * x + bias
    
    # Add RBF contributions
    for i in range(4):
        # Calculate distance from center
        dist_sq = (x - centers[i])**2
        # RBF: exp(-width * dist_sq)
        rbf_val = np.exp(-effective_widths[i] * dist_sq)
        pred += w[i] * rbf_val
        
    return pred
# EVOLVE-BLOCK-END
