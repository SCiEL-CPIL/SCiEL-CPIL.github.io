# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Mixed basis: global polynomial trend + local Gaussian RBFs
    # This allows the model to capture both smooth global structure
    # and localized features (peaks, valleys, sharp changes)
    
    # Global polynomial basis (4 terms for overall trend)
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x * y
    
    # Local Gaussian RBFs at key locations (6 terms for local detail)
    # Centers chosen to cover the domain [-1, 1]^2
    # Using different bandwidths for flexibility
    
    # Center of domain
    cx0, cy0 = 0.0, 0.0
    k0 = np.exp(-2.0 * ((x - cx0)**2 + (y - cy0)**2))
    
    # Corners
    cx1, cy1 = 1.0, 1.0
    k1 = np.exp(-2.0 * ((x - cx1)**2 + (y - cy1)**2))
    
    cx2, cy2 = 1.0, -1.0
    k2 = np.exp(-2.0 * ((x - cx2)**2 + (y - cy2)**2))
    
    cx3, cy3 = -1.0, 1.0
    k3 = np.exp(-2.0 * ((x - cx3)**2 + (y - cy3)**2))
    
    cx4, cy4 = -1.0, -1.0
    k4 = np.exp(-2.0 * ((x - cx4)**2 + (y - cy4)**2))
    
    # Edge midpoints for more coverage
    cx5, cy5 = 0.0, 1.0
    k5 = np.exp(-2.0 * ((x - cx5)**2 + (y - cy5)**2))
    
    # Combine: params[0:4] for polynomial, params[4:10] for RBFs
    pred = (params[0] * b0 + params[1] * b1 + params[2] * b2 + params[3] * b3 +
            params[4] * k0 + params[5] * k1 + params[6] * k2 +
            params[7] * k3 + params[8] * k4 + params[9] * k5)
    
    return pred
# EVOLVE-BLOCK-END
