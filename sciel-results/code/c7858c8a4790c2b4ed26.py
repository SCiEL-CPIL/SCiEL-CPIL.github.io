"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid RBF + polynomial basis for symbolic regression.
    
    Uses 4 Gaussian RBFs centered at a 2x2 grid (±0.5, ±0.5) with fixed width,
    plus 6 polynomial trend terms (1, x, y, x^2, y^2, x*y).
    
    f(x,y) = sum_{i=0}^{5} p_i * poly_i(x,y) + sum_{j=6}^{9} p_j * exp(-||[x,y]-c_j||^2 / (2*sigma^2))
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Polynomial trend terms (params 0-5)
    x2 = x * x
    y2 = y * y
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x2
    b4 = y2
    b5 = x * y
    
    # RBF terms (params 6-9): 4 centers on 2x2 grid
    sigma2 = 0.36  # sigma = 0.6, gives good spread
    centers = np.array([[0.5, 0.5],
                        [0.5, -0.5],
                        [-0.5, 0.5],
                        [-0.5, -0.5]])
    
    # Compute RBF values: exp(-d^2 / (2*sigma^2))
    dx = x[:, None] - centers[:, 0][None, :]  # shape (n, 4)
    dy = y[:, None] - centers[:, 1][None, :]  # shape (n, 4)
    d2 = dx * dx + dy * dy  # shape (n, 4)
    
    rbf_vals = np.exp(-d2 / (2.0 * sigma2))  # shape (n, 4)
    
    # Linear combination: poly part + RBF part
    pred = (params[0] * b0 +
            params[1] * b1 +
            params[2] * b2 +
            params[3] * b3 +
            params[4] * b4 +
            params[5] * b5 +
            params[6] * rbf_vals[:, 0] +
            params[7] * rbf_vals[:, 1] +
            params[8] * rbf_vals[:, 2] +
            params[9] * rbf_vals[:, 3])
    
    return pred
# EVOLVE-BLOCK-END
