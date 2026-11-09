import numpy as np
import time

# Module level time tracking for safety, though the evaluator controls the loop
_START_TIME = time.time()

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # x is expected to be in [0.1, 6]
    # Normalize to [-1, 1] for better numerical conditioning
    u = (x - 3.05) / 2.95
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(u) = 1
    # T_1(u) = u
    # T_n(u) = 2*u*T_{n-1}(u) - T_{n-2}(u)
    
    y = np.zeros_like(x)
    
    T_prev2 = np.ones_like(x)  # T_0
    y += params[0] * T_prev2
    
    if len(params) > 1:
        T_prev1 = u            # T_1
        y += params[1] * T_prev1
        
        for n in range(2, 10):
            T_curr = 2.0 * u * T_prev1 - T_prev2
            y += params[n] * T_curr
            T_prev2 = T_prev1
            T_prev1 = T_curr
            
    return y
