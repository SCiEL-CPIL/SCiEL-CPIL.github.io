import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: Chebyshev polynomial basis (degree 3) + Gaussian RBF terms.
    10 parameters total: 5 for polynomial basis, 5 for RBF weights.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Chebyshev polynomials of the first kind
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2.0 * x * x - 1.0
    T3_x = 4.0 * x * x * x - 3.0 * x
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2.0 * y * y - 1.0
    T3_y = 4.0 * y * y - 3.0 * y
    
    # Polynomial basis (5 terms)
    poly_pred = (p[0] * T0_x * T0_y + 
                 p[1] * T1_x * T0_y + 
                 p[2] * T0_x * T1_y + 
                 p[3] * T2_x * T0_y + 
                 p[4] * T0_x * T2_y)
    
    # RBF centers - chosen to cover the domain
    # Using 5 centers with distinct positions to avoid linear dependency
    centers = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [-0.5, -0.5],
        [0.5, -0.5],
        [-0.5, 0.5]
    ])
    
    # Gaussian RBF with beta=2.0 (sharper kernels)
    beta = 2.0
    
    rbf_pred = np.zeros_like(x)
    for i in range(5):
        cx, cy = centers[i]
        r2 = (x - cx) ** 2 + (y - cy) ** 2
        kernel = np.exp(-beta * r2)
        rbf_pred += p[5 + i] * kernel
    
    pred = poly_pred + rbf_pred
    
    # Safety check for finite values
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    # Deadline guard
    if time.time() > deadline:
        return pred
    
    return pred
