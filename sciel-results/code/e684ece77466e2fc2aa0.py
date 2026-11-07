import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model.
    Uses normalized coordinates to improve optimization conditioning.
    """
    # Normalization constants based on training data ranges
    X_SCALE = 0.5728
    V_SCALE = 0.2821
    
    # Handle potential division by zero if scales were 0, but they are not.
    X = x / X_SCALE
    V = v / V_SCALE
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Basis functions
    # We include terms up to cubic order in normalized variables.
    # This allows for quadratic and cubic nonlinearities.
    
    t1 = 1.0
    t2 = X
    t3 = X**2
    t4 = X**3
    t5 = V
    t6 = V**2
    t7 = V**3
    t8 = X * V
    t9 = X**2 * V
    t10 = X * V**2
    
    return (p0 * t1 + p1 * t2 + p2 * t3 + p3 * t4 + 
            p4 * t5 + p5 * t6 + p6 * t7 + p7 * t8 + 
            p8 * t9 + p9 * t10)
