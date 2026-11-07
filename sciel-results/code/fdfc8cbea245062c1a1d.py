"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = params[:10]
    
    # Radial basis function approach with fixed centers
    # Centers evenly spaced across [-3, 3]
    # We use 8 RBF centers + 2 polynomial terms (constant and linear)
    # This gives local approximation capability
    
    # Fixed centers for RBFs
    centers = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 2.5, -2.5])
    
    # Width parameter for Gaussian RBFs - fixed to avoid overfitting
    # sigma controls the width; smaller = more local
    sigma = 1.0
    
    # Compute RBF contributions
    result = np.zeros_like(x)
    
    # Polynomial terms for global trend
    result += p[0] * np.ones_like(x)
    result += p[1] * x
    
    # RBF terms
    for i in range(8):
        diff = x - centers[i]
        rbf = np.exp(-0.5 * (diff / sigma) ** 2)
        result += p[2 + i] * rbf
    
    return result
# EVOLVE-BLOCK-END
