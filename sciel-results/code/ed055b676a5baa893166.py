"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation using a single hidden-layer RBF network.
    
    The 10 parameters are:
    - params[0:9]: weights for 9 RBF centers
    - params[9]: bias term
    
    RBF centers are fixed at 9 evenly spaced points in [0.1, 6].
    The width (sigma) of the Gaussian RBFs is fixed to provide good coverage.
    """
    # Fixed RBF centers: 10 evenly spaced points in [0.1, 6]
    # We use 10 centers so that all 10 params are weights for the RBFs
    centers = np.linspace(0.1, 6.0, 10)
    
    # Fixed width for the Gaussian RBFs
    # Spacing is (6.0 - 0.1) / 9 ≈ 0.656
    # Sigma chosen to balance overlap and conditioning
    sigma = 0.4
    
    # Compute RBF activations: phi_i(x) = exp(-((x - c_i)^2) / (2 * sigma^2))
    # x has shape (n,), centers has shape (10,)
    # We need to compute for each x and each center
    x_expanded = x[:, np.newaxis]  # shape (n, 1)
    centers_expanded = centers[np.newaxis, :]  # shape (1, 10)
    
    # Squared distances
    diff = x_expanded - centers_expanded  # shape (n, 10)
    rbf_values = np.exp(-(diff ** 2) / (2.0 * sigma ** 2))  # shape (n, 10)
    
    # Output: weighted sum of RBF values
    # All 10 parameters are used as weights
    weights = params  # shape (10,)
    
    prediction = rbf_values @ weights
    
    return prediction
# EVOLVE-BLOCK-END
