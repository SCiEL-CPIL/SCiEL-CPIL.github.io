import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Tensor-product Fourier basis with multiple frequencies.
    Uses 10 parameters to form a smooth periodic-like function on [-1,1]^2.
    
    Basis functions:
    p0: 1 (bias)
    p1: cos(pi*x)
    p2: sin(pi*x)
    p3: cos(pi*y)
    p4: sin(pi*y)
    p5: cos(2*pi*x)
    p6: sin(2*pi*x)
    p7: cos(2*pi*y)
    p8: sin(2*pi*y)
    p9: cos(pi*x)*cos(pi*y)  (interaction term)
    """
    # Start time for deadline guard
    t_start = time.time()
    
    # Precompute basis functions
    cos_pi_x = np.cos(np.pi * x)
    sin_pi_x = np.sin(np.pi * x)
    cos_pi_y = np.cos(np.pi * y)
    sin_pi_y = np.sin(np.pi * y)
    cos_2pi_x = np.cos(2.0 * np.pi * x)
    sin_2pi_x = np.sin(2.0 * np.pi * x)
    cos_2pi_y = np.cos(2.0 * np.pi * y)
    sin_2pi_y = np.sin(2.0 * np.pi * y)
    
    # Interaction term
    interaction = cos_pi_x * cos_pi_y
    
    # Construct prediction
    pred = (params[0] * 1.0 +
            params[1] * cos_pi_x +
            params[2] * sin_pi_x +
            params[3] * cos_pi_y +
            params[4] * sin_pi_y +
            params[5] * cos_2pi_x +
            params[6] * sin_2pi_x +
            params[7] * cos_2pi_y +
            params[8] * sin_2pi_y +
            params[9] * interaction)
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.zeros_like(x)
    
    return pred
