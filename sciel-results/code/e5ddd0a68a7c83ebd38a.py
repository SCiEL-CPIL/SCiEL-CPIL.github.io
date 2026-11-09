import numpy as np
import time

_START_TIME = time.time()

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Normalize x to [-1, 1] for better conditioning
    # Domain is [0.1, 6], so center is 3.05, half-range is 2.95
    u = (x - 3.05) / 2.95
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(u) = 1
    # T_1(u) = u
    # T_n(u) = 2*u*T_{n-1}(u) - T_{n-2}(u)
    
    y = np.zeros_like(x)
    
    T_prev2 = np.ones_like(x)  # T_0
    y += params[0] * T_prev2
    
    T_prev1 = u                # T_1
    y += params[1] * T_prev1
    
    for n in range(2, 10):
        T_curr = 2.0 * u * T_prev1 - T_prev2
        y += params[n] * T_curr
        T_prev2 = T_prev1
        T_prev1 = T_curr
        
    return y
