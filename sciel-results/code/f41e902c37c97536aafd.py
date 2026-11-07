"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator.
    Uses normalized inputs and a mix of polynomial + nonlinear terms
    to capture the dynamics. 10 parameters total.
    
    Model terms (in order of params):
    p[0]: constant
    p[1]: x (normalized)
    p[2]: v (normalized)
    p[3]: x^2
    p[4]: x*v
    p[5]: v^2
    p[6]: sin(x)
    p[7]: tanh(x)
    p[8]: x^3
    p[9]: v^3
    """
    p = params
    
    # Normalize inputs to O(1) scale
    x_scale = 0.6 
    v_scale = 0.3
    
    xn = x / x_scale
    vn = v / v_scale
    
    # Precompute powers
    xn2 = xn * xn
    xn3 = xn2 * xn
    vn2 = vn * vn
    vn3 = vn2 * vn
    
    # Nonlinear terms
    sin_x = np.sin(xn)
    tanh_x = np.tanh(xn)
    
    return (
        p[0] +
        p[1] * xn +
        p[2] * vn +
        p[3] * xn2 +
        p[4] * xn * vn +
        p[5] * vn2 +
        p[6] * sin_x +
        p[7] * tanh_x +
        p[8] * xn3 +
        p[9] * vn3
    )
# EVOLVE-BLOCK-END
