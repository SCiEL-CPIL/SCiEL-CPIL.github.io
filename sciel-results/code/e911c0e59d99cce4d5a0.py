import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: Linear term + 3 Gaussian RBFs.
    Params:
    p[0]: slope
    p[1], p[2], p[3]: a1, c1_raw, w1_raw
    p[4], p[5], p[6]: a2, c2_raw, w2_raw
    p[7], p[8], p[9]: a3, c3_raw, w3_raw
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.array([])
        
    m = params[0]
    result = m * x
    
    # 3 Gaussians
    # Offsets to break symmetry at init (params=1)
    # tanh(1 + offset)
    # offsets: -1, 0, 1
    offsets = np.array([-1.0, 0.0, 1.0])
    
    for i in range(3):
        if time.time() > deadline:
            break
            
        a = params[1 + 3*i]
        c_raw = params[2 + 3*i]
        w_raw = params[3 + 3*i]
        
        c = 3.0 * np.tanh(c_raw + offsets[i])
        w = np.exp(w_raw)
        w = np.maximum(w, 1e-3)
        
        diff = x - c
        exponent = -0.5 * (diff / w) ** 2
        gauss = np.exp(exponent)
        
        result += a * gauss
        
    return result
