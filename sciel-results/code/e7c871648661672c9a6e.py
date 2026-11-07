import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 140.0

def equation(b, s, temp, pH, params):
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback to a simple mean-like prediction if timeout is imminent
        # This is unlikely for a single vectorized op, but satisfies the guard requirement
        return np.zeros_like(b, dtype=float)
        
    p = params
    
    # Normalize inputs to [-1, 1] based on training ranges
    # b: [1, 4.998]
    # s: [1, 4.999]
    # temp: [1.002, 39.98]
    # pH: [2, 12]
    
    b_norm = 2 * (b - 1.0) / (4.998 - 1.0) - 1
    s_norm = 2 * (s - 1.0) / (4.999 - 1.0) - 1
    temp_norm = 2 * (temp - 1.002) / (39.98 - 1.002) - 1
    ph_norm = 2 * (pH - 2.0) / (12.0 - 2.0) - 1
    
    # Clip to [-1, 1] to ensure stability during optimization
    b_norm = np.clip(b_norm, -1, 1)
    s_norm = np.clip(s_norm, -1, 1)
    temp_norm = np.clip(temp_norm, -1, 1)
    ph_norm = np.clip(ph_norm, -1, 1)
    
    # Basis Functions
    # Linear terms
    b_L = b_norm
    s_L = s_norm
    T_L = temp_norm
    pH_L = ph_norm
    
    # Quadratic terms for curvature (bell-shape for T/pH, saturation for s)
    T_Q = temp_norm**2
    pH_Q = ph_norm**2
    s_Q = s_norm**2
    
    # Interaction terms
    # T * pH: Temperature and pH often have synergistic or antagonistic effects on enzyme activity
    T_pH_I = temp_norm * ph_norm
    # T * s: Temperature can affect the efficiency of substrate utilization
    T_s_I = temp_norm * s_norm
    
    result = (p[0] 
              + p[1] * b_L 
              + p[2] * s_L 
              + p[3] * T_L 
              + p[4] * pH_L 
              + p[5] * T_Q 
              + p[6] * pH_Q 
              + p[7] * s_Q 
              + p[8] * T_pH_I 
              + p[9] * T_s_I)
              
    return result
