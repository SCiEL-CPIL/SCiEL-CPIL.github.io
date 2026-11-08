import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator.
    Structure: Duffing (x, x^3) + Van der Pol (v, x^2*v) + Cubic Damping (v^3) + Cross terms + Offset.
    """
    p = params
    
    # Precompute powers
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    # Terms
    t1 = x          # Linear spring
    t2 = x3         # Cubic spring (Duffing)
    t3 = v          # Linear damping
    t4 = v3         # Cubic damping
    t5 = x2 * v     # Van der Pol nonlinearity
    t6 = np.ones_like(x) # Offset
    t7 = x * v      # Cross coupling
    t8 = x2         # Quadratic spring
    t9 = v2         # Quadratic damping
    t10 = x3 * v    # Higher order coupling
    
    return (p[0] * t1 + 
            p[1] * t2 + 
            p[2] * t3 + 
            p[3] * t4 + 
            p[4] * t5 + 
            p[5] * t6 + 
            p[6] * t7 + 
            p[7] * t8 + 
            p[8] * t9 + 
            p[9] * t10)
