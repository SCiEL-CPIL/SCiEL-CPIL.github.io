import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 110.0 # Safety margin for 120s limit

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Additive model with tanh non-linearities.
    params: [bias, w1, w2, w3, w4, w5, w6, w7, w8, w9]
    y = bias 
      + w1 * tanh(w2*x + w3*v) 
      + w4 * tanh(w5*t + w6*x) 
      + w7 * tanh(w8*t + w9*v)
    """
    # Deadline guard (compliance with directive, though this function is fast)
    if time.time() - _START_TIME > _TIME_LIMIT:
         # If we hit the limit, return a safe fallback (e.g., just bias or zeros)
         # In practice, this shouldn't happen for a single vectorized op.
         return np.zeros_like(t)

    p = params
    # Extract parameters
    bias = p[0]
    w1, w2, w3 = p[1], p[2], p[3]
    w4, w5, w6 = p[4], p[5], p[6]
    w7, w8, w9 = p[7], p[8], p[9]
    
    # Compute non-linear terms
    term1 = w1 * np.tanh(w2 * x + w3 * v)
    term2 = w4 * np.tanh(w5 * t + w6 * x)
    term3 = w7 * np.tanh(w8 * t + w9 * v)
    
    result = bias + term1 + term2 + term3
    
    return result
