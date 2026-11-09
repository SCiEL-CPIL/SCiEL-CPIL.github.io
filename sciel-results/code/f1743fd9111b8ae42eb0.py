import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Bivariate Chebyshev basis of total degree <= 3, with L2-normalized basis
    # functions to improve conditioning for BFGS optimization.
    
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2.0 * x * x - 1.0
    T3_x = 4.0 * x * x * x - 3.0 * x
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2.0 * y * y - 1.0
    T3_y = 4.0 * y * y * y - 3.0 * y
    
    # Build the 10 basis functions
    b0 = T0_x * T0_y
    b1 = T1_x * T0_y
    b2 = T0_x * T1_y
    b3 = T2_x * T0_y
    b4 = T1_x * T1_y
    b5 = T0_x * T2_y
    b6 = T3_x * T0_y
    b7 = T2_x * T1_y
    b8 = T1_x * T2_y
    b9 = T0_x * T3_y
    
    # Stack into matrix for efficient computation
    B = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    # Compute L2 norms of each basis column over the training points
    # and normalize to unit norm to improve conditioning
    norms = np.sqrt(np.sum(B * B, axis=0))
    norms = np.where(norms > 1e-12, norms, 1.0)  # avoid division by zero
    B_norm = B / norms
    
    # Linear combination with normalized basis
    pred = B_norm @ params
    
    # Check for non-finite values
    if not np.all(np.isfinite(pred)):
        return np.zeros_like(y)
        
    return pred
