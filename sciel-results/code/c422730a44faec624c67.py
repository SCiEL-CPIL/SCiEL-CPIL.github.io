"""Symbolic regression with a structured basis: polynomial, trigonometric, and localized terms."""
import numpy as np

# EVOLVE-BLOCK-START

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict y from x using a linear combination of 10 basis functions.
    
    The basis is designed to span a rich function space while remaining well-conditioned
    for BFGS optimization starting from all ones.
    
    Basis functions (where t = x / 2, so t in [-1, 1]):
    0: 1
    1: t
    2: t^2
    3: t^3
    4: t^4
    5: sin(pi * t)
    6: cos(pi * t)
    7: sin(2 * pi * t)
    8: cos(2 * pi * t)
    9: t * sin(pi * t)  (interaction term to capture asymmetric oscillatory behavior)
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    
    # Ensure exactly 10 parameters
    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size), constant_values=0.0)
    else:
        p = p[:10]
        
    # Normalize x to [-1, 1] for better conditioning of polynomial basis
    t = x / 2.0
    
    # Construct basis functions
    b0 = np.ones_like(t)
    b1 = t
    b2 = t * t
    b3 = b2 * t
    b4 = b2 * b2
    
    # Trig terms using pi * t to cover the range
    pi_t = np.pi * t
    b5 = np.sin(pi_t)
    b6 = np.cos(pi_t)
    b7 = np.sin(2.0 * pi_t)
    b8 = np.cos(2.0 * pi_t)
    b9 = t * np.sin(pi_t)
    
    # Combine linearly
    result = (p[0] * b0 + 
              p[1] * b1 + 
              p[2] * b2 + 
              p[3] * b3 + 
              p[4] * b4 + 
              p[5] * b5 + 
              p[6] * b6 + 
              p[7] * b7 + 
              p[8] * b8 + 
              p[9] * b9)
              
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result

# EVOLVE-BLOCK-END
