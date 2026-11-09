"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the hidden function using a Radial Basis Function (RBF) network
    with fixed Gaussian centers spanning the input domain [0.1, 6].
    
    Uses 9 fixed centers and 1 learnable width parameter (params[9]).
    The width is parameterized as sigma = 0.05 + 5.0 * sigmoid(params[9])
    to keep it in a reasonable range and ensure smoothness for BFGS.
    """
    x = np.asarray(x, dtype=float)
    
    # 9 fixed centers evenly spaced across [0.1, 6]
    centers = np.linspace(0.1, 6.0, 9)
    
    # Learnable width: sigma = 0.05 + 5.0 * sigmoid(params[9])
    # This ensures sigma is in (0.05, 5.05), keeping it positive and bounded
    sigma = 0.05 + 5.0 / (1.0 + np.exp(-np.clip(params[9], -50, 50)))
    
    # Compute RBF values: sum of Gaussian kernels
    # y = sum_{i=0}^{8} params[i] * exp(-((x - centers[i])^2) / (2 * sigma^2))
    y = np.zeros_like(x)
    
    for i in range(9):
        diff = x - centers[i]
        y += params[i] * np.exp(-(diff ** 2) / (2.0 * sigma ** 2))
    
    return y
# EVOLVE-BLOCK-END
