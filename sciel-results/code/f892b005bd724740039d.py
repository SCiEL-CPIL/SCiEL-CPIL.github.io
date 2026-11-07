"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: 9 Gaussian RBFs + 1 constant term.
    Fixed centers and widths.
    Linear in parameters.
    """
    # Centers spread across the domain [-2, 2]
    centers = np.linspace(-2.0, 2.0, 9)
    # Narrower width to allow local fitting
    sigma = 0.5
    
    # Compute RBF values
    # x shape: (N,), centers shape: (9,)
    # diff shape: (N, 9)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Add constant term as the 10th basis function
    # Basis shape: (N, 10)
    basis = np.hstack((rbf, np.ones((len(x), 1))))
    
    # result = sum_{i=0}^{9} params[i] * basis_i(x)
    result = basis @ params
    return result
# EVOLVE-BLOCK-END
