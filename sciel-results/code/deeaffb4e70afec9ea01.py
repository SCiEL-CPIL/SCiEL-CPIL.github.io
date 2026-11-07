"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Fixed scaling constants based on training ranges
# x range: [-0.5728, 0.5275], center ~ -0.02265, half-range ~ 0.55015
# v range: [-0.1754, 0.2821], center ~ 0.05335, half-range ~ 0.22875
X_CENTER = -0.02265
X_SCALE = 0.55015
V_CENTER = 0.05335
V_SCALE = 0.22875

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Acceleration of a damped nonlinear oscillator with driving force, from position and velocity."""
    # Scale inputs to O(1) for better conditioning
    xs = (x - X_CENTER) / X_SCALE
    vs = (v - V_CENTER) / V_SCALE
    
    # Build a rich basis for a nonlinear oscillator (Duffing-like)
    # Basis: 1, xs, vs, xs^2, xs^3, xs^4, vs^2, xs*vs, xs^2*vs, vs^3
    # This gives 10 terms matching the 10 params
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    result = (
        p0 * np.ones_like(xs) +
        p1 * xs +
        p2 * vs +
        p3 * xs**2 +
        p4 * xs**3 +
        p5 * xs**4 +
        p6 * vs**2 +
        p7 * xs * vs +
        p8 * xs**2 * vs +
        p9 * vs**3
    )
    
    return result
# EVOLVE-BLOCK-END
