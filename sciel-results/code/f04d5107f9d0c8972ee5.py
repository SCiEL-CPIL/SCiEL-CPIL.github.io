import numpy as np
import time

_START_TIME = time.time()

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Safety: ensure x is in a reasonable domain for log/exp
    x_safe = np.clip(x, 1e-8, None)
    
    # Compute basis functions
    # 0: 1
    b0 = np.ones_like(x)
    
    # 1: x
    b1 = x
    
    # 2: x^2
    b2 = x * x
    
    # 3: x^3
    b3 = b2 * x
    
    # 4: exp(x) - clipped to avoid overflow
    exp_x = np.exp(np.clip(x, -50, 50))
    
    # 5: exp(-x)
    exp_neg_x = np.exp(np.clip(-x, -50, 50))
    
    # 6: log(x)
    log_x = np.log(x_safe)
    
    # 7: sin(x)
    sin_x = np.sin(x)
    
    # 8: cos(x)
    cos_x = np.cos(x)
    
    # 9: 1/x
    inv_x = 1.0 / x_safe
    
    # Combine
    y = (params[0] * b0 + 
         params[1] * b1 + 
         params[2] * b2 + 
         params[3] * b3 + 
         params[4] * exp_x + 
         params[5] * exp_neg_x + 
         params[6] * log_x + 
         params[7] * sin_x + 
         params[8] * cos_x + 
         params[9] * inv_x)
    
    return y
