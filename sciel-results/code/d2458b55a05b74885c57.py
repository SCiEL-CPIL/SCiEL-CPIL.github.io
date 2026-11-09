"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard: ensure we return within 150s wall-clock
    t_start = time.time()
    deadline = 145.0  # 5s safety margin
    
    # Hybrid RBF + Polynomial Basis
    # 5 RBFs centered at (0,0) and the four corners of [-1, 1]^2
    # 5 Polynomial terms to capture global trends
    
    # RBF parameters
    sigma = 0.7
    inv_2sigma2 = 1.0 / (2.0 * sigma**2)
    
    # Compute RBFs
    rbf0 = np.exp(-(x**2 + y**2) * inv_2sigma2)  # origin
    rbf1 = np.exp(-((x - 1.0)**2 + (y - 1.0)**2) * inv_2sigma2)  # (1,1)
    rbf2 = np.exp(-((x + 1.0)**2 + (y + 1.0)**2) * inv_2sigma2)  # (-1,-1)
    rbf3 = np.exp(-((x - 1.0)**2 + (y + 1.0)**2) * inv_2sigma2)  # (1,-1)
    rbf4 = np.exp(-((x + 1.0)**2 + (y - 1.0)**2) * inv_2sigma2)  # (-1,1)
    
    # Compute polynomial terms
    poly0 = np.ones_like(x)  # 1
    poly1 = x
    poly2 = y
    poly3 = x * y
    poly4 = x**2 + y**2
    
    # Stack basis functions into a design matrix (n_points x 10)
    B = np.column_stack([
        rbf0, rbf1, rbf2, rbf3, rbf4,
        poly0, poly1, poly2, poly3, poly4
    ])
    
    # Matrix-vector product with safety against non-finite values
    with np.errstate(over='ignore', invalid='ignore'):
        pred = B @ params
    
    # Replace any non-finite values with 0 to keep BFGS stable
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
