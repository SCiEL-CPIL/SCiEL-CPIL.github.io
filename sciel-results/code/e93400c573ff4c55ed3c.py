"""Rewritten program for symbolic regression with 3 inputs."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 150.0
_TIME_MARGIN = 5.0

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model with linear, sinusoidal, and interaction terms.
    
    The previous quadratic model scored ~7.39, suggesting either the target is not
    well-approximated by a quadratic polynomial or the optimizer struggled.
    
    This model uses:
    - Linear terms (x, y, z)
    - Sinusoidal terms (sin(x), sin(y), sin(z)) to capture periodic/non-linear curvature
    - Interaction terms (xy, yz, zx) to capture multiplicative effects
    
    Inputs are assumed to be in [0.5, 2].
    """
    if time.time() - _START_TIME > _TIME_LIMIT - _TIME_MARGIN:
        # Fallback to simple linear if time is running out
        return params[0] + params[1] * x + params[2] * y + params[3] * z

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    
    p = params
    
    # Basis functions
    # 1. Bias
    b0 = 1.0
    # 2-4. Linear
    b1 = x
    b2 = y
    b3 = z
    # 5-7. Sinusoidal (scaled to be in a reasonable range, sin of [0.5, 2] is fine)
    b4 = np.sin(x)
    b5 = np.sin(y)
    b6 = np.sin(z)
    # 8-10. Interactions
    b7 = x * y
    b8 = y * z
    b9 = z * x
    
    pred = (p[0] * b0 +
            p[1] * b1 +
            p[2] * b2 +
            p[3] * b3 +
            p[4] * b4 +
            p[5] * b5 +
            p[6] * b6 +
            p[7] * b7 +
            p[8] * b8 +
            p[9] * b9)
            
    return pred
# EVOLVE-BLOCK-END
