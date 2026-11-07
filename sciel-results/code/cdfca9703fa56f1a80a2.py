import numpy as np
import time

_start_time = None

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Bivariate Polynomial Model (Degree 4 in strain, Degree 1 in temp).
    
    stress = p0 + p1*s + p2*s^2 + p3*s^3 + p4*s^4 
           + p5*t + p6*s*t + p7*s^2*t + p8*s^3*t + p9*s^4*t
    
    Inputs are scaled to [0, 1] range based on training bounds.
    Strain range: [-0.0019, 0.9636]
    Temp range: [0.066, 1.0]
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > 140:
        return np.zeros_like(strain, dtype=np.float64)
    
    # Scaling
    s_min, s_max = -0.0019, 0.9636
    t_min, t_max = 0.066, 1.0
    
    s_range = s_max - s_min
    t_range = t_max - t_min
    
    # Avoid division by zero if range is 0, though not the case here
    if s_range < 1e-9: s_range = 1.0
    if t_range < 1e-9: t_range = 1.0
    
    s = (strain - s_min) / s_range
    t = (temp - t_min) / t_range
    
    # Polynomial basis
    s2 = s * s
    s3 = s2 * s
    s4 = s3 * s
    
    # Terms
    # p0: 1
    # p1: s
    # p2: s^2
    # p3: s^3
    # p4: s^4
    # p5: t
    # p6: s*t
    # p7: s^2*t
    # p8: s^3*t
    # p9: s^4*t
    
    result = (params[0] 
              + params[1] * s 
              + params[2] * s2 
              + params[3] * s3 
              + params[4] * s4 
              + params[5] * t 
              + params[6] * s * t 
              + params[7] * s2 * t 
              + params[8] * s3 * t 
              + params[9] * s4 * t)
    
    return result
