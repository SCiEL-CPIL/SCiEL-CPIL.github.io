# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Radial Basis Function (RBF) expansion with fixed centers and width.
    Basis: [1, exp(-beta*(x-c_1)^2), ..., exp(-beta*(x-c_9)^2)]
    Centers are evenly spaced in [-2, 2].
    This approach is better for functions with local features or non-polynomial shapes
    that a global polynomial of degree 9 might fail to capture.
    """
    x = np.asarray(x, dtype=float)
    p = params
    
    # Define 9 fixed centers
    centers = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    
    # Fixed width parameter for the Gaussian RBFs
    # A beta of 1.0 to 4.0 is typical. 2.0 provides a balance between local and global.
    beta = 2.0
    
    # Compute RBF values
    # x is shape (N,), centers is shape (9,)
    # We need (N, 9) matrix of distances
    # x[:, None] - centers[None, :]
    diff = x[:, None] - centers[None, :]
    
    # Gaussian kernel
    rbf_values = np.exp(-beta * diff**2)
    
    # Combine with parameters
    # p[0] is the constant term (bias)
    # p[1:] are the weights for the 9 RBFs
    result = p[0] + np.sum(p[1:] * rbf_values, axis=1)
    
    return result
# EVOLVE-BLOCK-END
