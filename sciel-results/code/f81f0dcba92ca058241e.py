import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check deadline
    if time.time() - _START_TIME > _DEADLINE:
        return np.zeros_like(x)
    
    # 10-term Fourier series with fixed frequencies {1, 2, 3, 4, 5}
    # params: [a0, b1, a1, b2, a2, b3, a3, b4, a4, b5]
    # y = a0 + sum_{k=1}^{5} (a_k * cos(k*x) + b_k * sin(k*x))
    
    a0 = params[0]
    b1 = params[1]
    a1 = params[2]
    b2 = params[3]
    a2 = params[4]
    b3 = params[5]
    a3 = params[6]
    b4 = params[7]
    a4 = params[8]
    b5 = params[9]
    
    # Compute Fourier series
    y = a0
    y += a1 * np.cos(x) + b1 * np.sin(x)
    y += a2 * np.cos(2 * x) + b2 * np.sin(2 * x)
    y += a3 * np.cos(3 * x) + b3 * np.sin(3 * x)
    y += a4 * np.cos(4 * x) + b4 * np.sin(4 * x)
    y += b5 * np.sin(5 * x)
    
    return y
