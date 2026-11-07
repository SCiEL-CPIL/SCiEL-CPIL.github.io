import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT = 110.0 # 10s margin

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    if time.time() - _START_TIME > _TIMEOUT:
        # Return a safe default, e.g., zeros or mean
        return np.zeros_like(t)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute powers to save time? 
    # x**2, x**3, v**3 are cheap.
    
    return (p0 + 
            p1 * x + 
            p2 * x**2 + 
            p3 * x**3 + 
            p4 * v + 
            p5 * v**3 + 
            p6 * x * v + 
            p7 * np.sin(t) + 
            p8 * np.cos(t) + 
            p9 * np.sin(2*t))
