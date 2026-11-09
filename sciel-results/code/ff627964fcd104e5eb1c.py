import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with driving force, from position and velocity.
    Uses a normalized polynomial basis to capture Duffing-like stiffness and nonlinear damping.
    """
    # Normalization constants derived from input ranges
    # x: [-0.5728, 0.5275] -> center -0.02265, scale 0.6
    # v: [-0.1754, 0.2821] -> center 0.05335, scale 0.25
    
    x_center = -0.02265
    x_scale = 0.6
    v_center = 0.05335
    v_scale = 0.25
    
    X = (x - x_center) / x_scale
    V = (v - v_center) / v_scale
    
    # Basis functions
    b0 = np.ones_like(X)
    b1 = X
    b2 = X**3
    b3 = V
    b4 = V**3
    b5 = X * V
    b6 = X**2
    b7 = V**2
    b8 = X**5
    b9 = V**5
    
    # Combine with parameters
    y = (params[0] * b0 +
         params[1] * b1 +
         params[2] * b2 +
         params[3] * b3 +
         params[4] * b4 +
         params[5] * b5 +
         params[6] * b6 +
         params[7] * b7 +
         params[8] * b8 +
         params[9] * b9)
         
    return y
