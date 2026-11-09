import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Core Duffing oscillator with nonlinear damping terms
    # Driving force (fundamental and phase-shifted)
    c1 = np.cos(t)
    s1 = np.sin(t)
    
    # Restoring force (linear, quadratic, cubic)
    x1 = x
    x2 = x**2
    x3 = x**3
    
    # Damping terms (linear, quadratic)
    v1 = v
    v2 = v**2
    
    # Cross/nonlinear damping terms
    xv = x * v
    x2v = x**2 * v
    
    # Constant term
    ones = np.ones_like(t)
    
    # Combine with fitted parameters
    # Order: c1, s1, x1, x2, x3, v1, v2, xv, x2v, ones
    return (params[0] * c1 + 
            params[1] * s1 + 
            params[2] * x1 + 
            params[3] * x2 + 
            params[4] * x3 + 
            params[5] * v1 + 
            params[6] * v2 + 
            params[7] * xv + 
            params[8] * x2v + 
            params[9] * ones)
