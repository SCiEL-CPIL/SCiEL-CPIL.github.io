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
    # Fixed RBF centers: 9 evenly spaced points in [0.1, 6]
    # We use 9 centers so that with 1 bias we use all 10 params
    centers = np.linspace(0.1, 6.0, 9)
    
    # Fixed width for the Gaussian RBFs
    # The spacing between centers is (6.0 - 0.1) / 8 = 0.7375
    # A good sigma is roughly half the spacing to allow overlap but not too much
    sigma = 0.4
    
    # Compute RBF activations: phi_i(x) = exp(-((x - c_i)^2) / (2 * sigma^2))
    # x has shape (n,), centers has shape (9,)
    # We need to compute for each x and each center
    x_expanded = x[:, np.newaxis]  # shape (n, 1)
    centers_expanded = centers[np.newaxis, :]  # shape (1, 9)
    
    # Squared distances
    diff = x_expanded - centers_expanded  # shape (n, 9)
    rbf_values = np.exp(-(diff ** 2) / (2.0 * sigma ** 2))  # shape (n, 9)
    
    # Output: weighted sum of RBF values + bias
    weights = params[0:9]  # shape (9,)
    bias = params[9]
    
    prediction = rbf_values @ weights + bias
    
    return prediction
# EVOLVE-BLOCK-END
