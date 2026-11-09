import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Structurally motivated nonlinear oscillator basis
    # Combines Duffing-type restoring force, Van der Pol damping, and driven terms
    
    # Drive terms: fundamental frequency (cos and sin for phase flexibility)
    c1 = np.cos(t)
    s1 = np.sin(t)
    
    # Position terms: linear, quadratic, cubic (Duffing oscillator)
    x1 = x
    x2 = x**2
    x3 = x**3
    
    # Velocity terms: linear damping and cubic damping (Van der Pol style)
    v1 = v
    v3 = v**3
    
    # Interaction terms: capture nonlinear coupling between position and velocity
    xv = x * v
    x2v = x**2 * v  # Van der Pol specific term
    
    # Constant term (bias)
    ones = np.ones_like(t)
    
    # Combine all terms with fitted parameters
    # Order: c1, s1, x1, x2, x3, v1, v3, xv, x2v, ones
    return (params[0] * c1 + 
            params[1] * s1 + 
            params[2] * x1 + 
            params[3] * x2 + 
            params[4] * x3 + 
            params[5] * v1 + 
            params[6] * v3 + 
            params[7] * xv + 
            params[8] * x2v + 
            params[9] * ones)
