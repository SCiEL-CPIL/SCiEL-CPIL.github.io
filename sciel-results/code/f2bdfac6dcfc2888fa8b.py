import numpy as np
import time

_START_TIME = time.time()

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    
    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size), constant_values=0.0)
    else:
        p = p[:10]
        
    # Deadline guard
    if time.time() - _START_TIME > 145.0:
        return np.zeros_like(x)

    # Unpack and transform parameters to bounded ranges
    # This helps BFGS by keeping parameters in a reasonable range
    # and preventing overflow/underflow issues.
    
    # Component 1: Gaussian 1
    A1 = 10.0 * np.tanh(p[0])
    c1 = 2.0 * np.tanh(p[1])
    w1 = 0.1 + 1.9 * (0.5 * (1.0 + np.tanh(p[2]))) # Range [0.1, 2.0]
    
    # Component 2: Gaussian 2
    A2 = 10.0 * np.tanh(p[3])
    c2 = 2.0 * np.tanh(p[4])
    w2 = 0.1 + 1.9 * (0.5 * (1.0 + np.tanh(p[5]))) # Range [0.1, 2.0]
    
    # Component 3: Gaussian 3
    A3 = 10.0 * np.tanh(p[6])
    c3 = 2.0 * np.tanh(p[7])
    w3 = 0.1 + 1.9 * (0.5 * (1.0 + np.tanh(p[8]))) # Range [0.1, 2.0]
    
    # Constant
    C = 10.0 * np.tanh(p[9])
    
    # Calculate Gaussians
    # Use safe division
    y1 = A1 * np.exp(-((x - c1) ** 2) / (2.0 * w1 * w1))
    y2 = A2 * np.exp(-((x - c2) ** 2) / (2.0 * w2 * w2))
    y3 = A3 * np.exp(-((x - c3) ** 2) / (2.0 * w3 * w3))
    
    result = y1 + y2 + y3 + C
    
    # Ensure finite
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
