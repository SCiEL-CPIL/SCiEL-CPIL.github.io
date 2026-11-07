import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation for symbolic regression with 2 inputs x, y.
    
    Uses a parametric trigonometric Fourier series with 10 basis functions.
    The basis includes low-frequency sine and cosine terms of x, y, and their
    combinations (x+y, x-y), plus a bias and a linear interaction term.
    
    This representation is linear in the 10 parameters, ensuring BFGS converges
    to the global minimum for this subspace. The trigonometric basis is well-suited
    for functions with periodic or smooth non-linear behavior on [-1, 1].
    
    Basis functions:
    1. 1 (bias)
    2. sin(pi*x)
    3. cos(pi*x)
    4. sin(pi*y)
    5. cos(pi*y)
    6. sin(pi*(x+y))
    7. cos(pi*(x-y))
    8. sin(2*pi*x)
    9. sin(2*pi*y)
    10. x*y (interaction term)
    """
    start_time = time.time()
    
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    
    # Ensure p has exactly 10 elements
    if p.shape[0] != 10:
        p = np.resize(p, 10)
    
    # Compute basis functions
    b1 = np.ones_like(x)
    b2 = np.sin(np.pi * x)
    b3 = np.cos(np.pi * x)
    b4 = np.sin(np.pi * y)
    b5 = np.cos(np.pi * y)
    b6 = np.sin(np.pi * (x + y))
    b7 = np.cos(np.pi * (x - y))
    b8 = np.sin(2.0 * np.pi * x)
    b9 = np.sin(2.0 * np.pi * y)
    b10 = x * y
    
    # Stack into matrix: shape (n_points, 10)
    B = np.column_stack([b1, b2, b3, b4, b5, b6, b7, b8, b9, b10])
    
    # Compute prediction: B @ params
    pred = B @ p
    
    # Check for non-finite values
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 110:
        return pred
        
    return pred
