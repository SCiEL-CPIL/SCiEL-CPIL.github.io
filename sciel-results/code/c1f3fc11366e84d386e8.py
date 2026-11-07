"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation using a single hidden-layer RBF network with 10 centers.
    
    The 10 parameters are:
    - params[0:10]: weights for 10 RBF centers
    
    RBF centers are fixed at 10 evenly spaced points in [0.1, 6].
    The width (sigma) of the Gaussian RBFs is fixed.
    """
    # Fixed RBF centers: 10 evenly spaced points in [0.1, 6]
    centers = np.linspace(0.1, 6.0, 10)
    
    # Fixed width for the Gaussian RBFs
    # Spacing is (6.0 - 0.1) / 9 ≈ 0.655
    # Sigma chosen to provide good overlap
    sigma = 0.35
    
    # Compute RBF activations: phi_i(x) = exp(-((x - c_i)^2) / (2 * sigma^2))
    x_expanded = x[:, np.newaxis]  # shape (n, 1)
    centers_expanded = centers[np.newaxis, :]  # shape (1, 10)
    
    # Squared distances
    diff = x_expanded - centers_expanded  # shape (n, 10)
    rbf_values = np.exp(-(diff ** 2) / (2.0 * sigma ** 2))  # shape (n, 10)
    
    # Output: weighted sum of RBF values
    weights = params  # shape (10,)
    
    prediction = rbf_values @ weights
    
    return prediction
# EVOLVE-BLOCK-END
