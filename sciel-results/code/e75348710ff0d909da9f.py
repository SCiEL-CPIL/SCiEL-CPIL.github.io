import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    t_start = time.time()
    deadline = 145.0
    
    # Basis functions
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x**2
    b4 = y**2
    b5 = x * y
    b6 = np.sin(x)
    b7 = np.sin(y)
    b8 = np.cos(x) - 1.0
    b9 = np.cos(y) - 1.0
    
    pred = (params[0] * b0 + params[1] * b1 + params[2] * b2 + 
            params[3] * b3 + params[4] * b4 + params[5] * b5 + 
            params[6] * b6 + params[7] * b7 + params[8] * b8 + 
            params[9] * b9)
    
    if time.time() - t_start > deadline:
        return pred
        
    return pred
