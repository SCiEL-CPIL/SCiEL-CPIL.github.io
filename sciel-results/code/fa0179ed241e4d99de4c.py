import numpy as np
from scipy.optimize import minimize
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a hybrid basis model.
    Combines a global polynomial basis (Chebyshev) for smooth trends
    with local Gaussian RBFs for capturing non-smooth or localized features.
    The model is a linear combination of 10 basis functions:
    1. Constant
    2. x
    3. y
    4. x^2
    5. y^2
    6. x*y
    7. RBF centered at (0, 0)
    8. RBF centered at (0.5, 0.5)
    9. RBF centered at (-0.5, -0.5)
    10. RBF centered at (0.5, -0.5)
    """
    # Compute polynomial basis terms
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x**2
    b4 = y**2
    b5 = x * y
    
    # Compute RBF terms
    # Using a fixed bandwidth sigma = 0.5 for stability
    sigma = 0.5
    
    # RBF 1: center (0, 0)
    r1_sq = x**2 + y**2
    b6 = np.exp(-r1_sq / (2 * sigma**2))
    
    # RBF 2: center (0.5, 0.5)
    r2_sq = (x - 0.5)**2 + (y - 0.5)**2
    b7 = np.exp(-r2_sq / (2 * sigma**2))
    
    # RBF 3: center (-0.5, -0.5)
    r3_sq = (x + 0.5)**2 + (y + 0.5)**2
    b8 = np.exp(-r3_sq / (2 * sigma**2))
    
    # RBF 4: center (0.5, -0.5)
    r4_sq = (x - 0.5)**2 + (y + 0.5)**2
    b9 = np.exp(-r4_sq / (2 * sigma**2))
    
    # Stack into matrix of shape (N, 10)
    B = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    # Prediction is B @ params
    pred = B.dot(params)
    
    return pred
