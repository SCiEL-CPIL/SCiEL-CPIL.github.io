"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model using tanh saturation functions.
    
    Structure:
    a = Drive + Nonlinear_Stiffness + Nonlinear_Damping + Cross_Terms + Offset
    
    Parameters:
    p0, p1, p2: Drive amplitude, frequency, phase
    p3, p4: Stiffness amplitude and nonlinearity scale (tanh(p4*x))
    p5, p6: Damping amplitude and nonlinearity scale (tanh(p6*v))
    p7: Cross-term coefficient for x*v
    p8: Quadratic stiffness coefficient for x^2
    p9: Quadratic damping coefficient for v^2
    p10: Static offset
    """
    # Check time budget
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        return np.zeros_like(x)

    p = params
    
    # Drive term
    drive = p[0] * np.sin(p[1] * t + p[2])
    
    # Nonlinear Stiffness: p3 * tanh(p4 * x)
    # If p4 is large, behaves like p3 * sign(x) (square wave) or p3*x if p4 small? 
    # Actually tanh(z) ~ z for small z, ~ 1 for large z.
    # So p3 * tanh(p4 * x) captures linear stiffness for small x, saturating for large x.
    stiff = p[3] * np.tanh(p[4] * x)
    
    # Nonlinear Damping: p5 * tanh(p6 * v)
    damp = p[5] * np.tanh(p[6] * v)
    
    # Cross term
    cross = p[7] * x * v
    
    # Quadratic terms (asymmetry)
    quad_x = p[8] * (x ** 2)
    quad_v = p[9] * (v ** 2)
    
    # Offset
    offset = p[10]
    
    # Sum all components
    return drive + stiff + damp + cross + quad_x + quad_v + offset
# EVOLVE-BLOCK-END
