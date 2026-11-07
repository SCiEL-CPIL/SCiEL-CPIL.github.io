import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    start_time = time.time()
    
    # Basis: Complete polynomial of degree 3 in x and v
    # x, v are raw inputs
    
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    # Terms:
    # 0: 1
    # 1: x
    # 2: v
    # 3: x^2
    # 4: x*v
    # 5: v^2
    # 6: x^3
    # 7: x^2*v
    # 8: x*v^2
    # 9: v^3
    
    pred = np.zeros_like(x, dtype=float)
    
    pred += params[0] * 1.0
    pred += params[1] * x
    pred += params[2] * v
    pred += params[3] * x2
    pred += params[4] * (x * v)
    pred += params[5] * v2
    pred += params[6] * x3
    pred += params[7] * (x2 * v)
    pred += params[8] * (x * v2)
    pred += params[9] * v3
    
    # Deadline guard (likely not needed for this simple calc, but required)
    elapsed = time.time() - start_time
    if elapsed > 149.0:
         # Return best effort (which is just pred)
         pass
        
    return pred
