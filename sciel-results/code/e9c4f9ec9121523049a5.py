"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-informed basis for a nonlinear oscillator (Duffing-Van der Pol type).
    
    The model assumes the acceleration y is a function of position x, velocity v, and time t.
    y = f(x, v, t)
    
    Basis terms:
    1. Bias
    2. Linear position (spring)
    3. Cubic position (Duffing)
    4. Linear velocity (damping)
    5. Cubic velocity (Van der Pol)
    6. Position-velocity cross term
    7. Sin(t) forcing
    8. Cos(t) forcing
    9. Sin(2t) forcing (2nd harmonic)
    10. Cos(2t) forcing (2nd harmonic)
    
    Inputs are scaled to O(1) to improve BFGS convergence.
    """
    start_time = time.time()
    
    # Unpack parameters
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # Scale inputs to roughly [-1, 1] range for better conditioning
    # x range: [-0.34, 0.36] -> scale ~ 0.35
    # v range: [-0.21, 0.17] -> scale ~ 0.20
    scale_x = 0.35
    scale_v = 0.20
    
    xs = x / scale_x
    vs = v / scale_v
    
    # Precompute powers
    xs2 = xs * xs
    vs2 = vs * vs
    xs3 = xs2 * xs
    vs3 = vs2 * vs
    
    # Basis terms
    b0 = 1.0
    b1 = xs
    b2 = xs3
    b3 = vs
    b4 = vs3
    b5 = xs * vs
    b6 = np.sin(t)
    b7 = np.cos(t)
    b8 = np.sin(2.0 * t)
    b9 = np.cos(2.0 * t)
    
    # Combine linear combination
    result = (p0 * b0 + 
              p1 * b1 + 
              p2 * b2 + 
              p3 * b3 + 
              p4 * b4 + 
              p5 * b5 + 
              p6 * b6 + 
              p7 * b7 + 
              p8 * b8 + 
              p9 * b9)
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(result)):
        return np.zeros_like(t)
        
    return result
# EVOLVE-BLOCK-END
