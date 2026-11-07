import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation for symbolic regression with 2 inputs x, y.
    Combines polynomial and tanh-based bounded non-linearities.
    The 10 parameters correspond to the weights of these basis functions.
    
    Basis functions:
    0: 1 (bias)
    1: x
    2: y
    3: x * y
    4: x * x
    5: y * y
    6: tanh(2.0 * x)
    7: tanh(2.0 * y)
    8: tanh(2.0 * (x + y))
    9: tanh(2.0 * (x - y))
    """
    # Deadline guard
    t0 = time.time()
    DEADLINE = 140.0  # 10s margin before 150s kill
    
    # Ensure inputs are float64 arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Check for non-finite inputs (safety)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(np.isfinite(p))):
        return np.zeros_like(x)
    
    # Compute basis functions
    # Polynomial basis (low order for stability)
    b1 = np.ones_like(x)
    b2 = x
    b3 = y
    b4 = x * y
    b5 = x * x
    b6 = y * y
    
    # Tanh-based bounded non-linearities
    # These provide flexible non-linear capacity while being well-conditioned
    t1 = np.tanh(2.0 * x)
    t2 = np.tanh(2.0 * y)
    t3 = np.tanh(2.0 * (x + y))
    t4 = np.tanh(2.0 * (x - y))
    
    # Build prediction as linear combination
    # Use all 10 params
    pred = (p[0] * b1 + 
            p[1] * b2 + 
            p[2] * b3 + 
            p[3] * b4 + 
            p[4] * b5 + 
            p[5] * b6 + 
            p[6] * t1 + 
            p[7] * t2 + 
            p[8] * t3 + 
            p[9] * t4)
    
    # Check deadline
    if time.time() - t0 > DEADLINE:
        return pred
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
