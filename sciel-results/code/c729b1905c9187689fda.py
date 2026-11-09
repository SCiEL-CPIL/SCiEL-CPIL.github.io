"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with time-dependent driving force.
    Structural change: Includes cubic nonlinearity, quadratic damping, periodic driving,
    and parametric excitation terms.
    """
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # p0: bias
    # p1: linear stiffness
    # p2: cubic stiffness (Duffing)
    # p3: linear damping
    # p4: quadratic damping
    # p5: sine drive
    # p6: cosine drive
    # p7: linear time trend
    # p8: parametric excitation (x * sin(t))
    # p9: parametric excitation (x * cos(t))
    
    return (params[0] +
            params[1] * x +
            params[2] * x3 +
            params[3] * v +
            params[4] * v2 +
            params[5] * sin_t +
            params[6] * cos_t +
            params[7] * t +
            params[8] * x * sin_t +
            params[9] * x * cos_t)
# EVOLVE-BLOCK-END
