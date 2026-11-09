"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Damped nonlinear oscillator (Duffing-type) with sinusoidal driving.
    
    a = p0 + p1*x + p2*v + p3*x^3 + p4*v^2 + p5*sin(t) + p6*cos(t) + p7*x*v + p8*x^2 + p9*sin(2t)
    
    Uses all 10 parameters. The model captures:
    - p0: bias/offset
    - p1: linear spring constant
    - p2: linear damping
    - p3: cubic nonlinearity (Duffing)
    - p4: quadratic velocity damping
    - p5, p6: sinusoidal driving (fundamental)
    - p7: cross term x*v (Gyroscopic/magnetic coupling)
    - p8: quadratic position (softening/hardening)
    - p9: second harmonic driving
    """
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    xv = x * v
    
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    sin_2t = np.sin(2 * t)
    
    return (params[0] +
            params[1] * x +
            params[2] * v +
            params[3] * x3 +
            params[4] * v2 +
            params[5] * sin_t +
            params[6] * cos_t +
            params[7] * xv +
            params[8] * x2 +
            params[9] * sin_2t)
# EVOLVE-BLOCK-END
