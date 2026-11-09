"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Duffing-type nonlinear oscillator with nonlinear damping and parametric drive.
    
    Structure:
      a = p0 + p1*x + p2*x^3 + p3*v + p4*v*|v| + p5*sin(t) + p6*cos(t)
          + p7*sin(t)*x + p8*cos(t)*x + p9*x*v
    
    This captures:
      - Linear + cubic restoring force (Duffing)
      - Linear + quadratic (nonlinear) damping
      - External periodic drive
      - Parametric excitation (time-modified stiffness)
      - Velocity-position coupling
    """
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    x2 = x * x
    x3 = x2 * x
    abs_v = np.abs(v)
    v_abs_v = v * abs_v
    xv = x * v
    
    return (params[0] +
            params[1] * x +
            params[2] * x3 +
            params[3] * v +
            params[4] * v_abs_v +
            params[5] * sin_t +
            params[6] * cos_t +
            params[7] * sin_t * x +
            params[8] * cos_t * x +
            params[9] * xv)
# EVOLVE-BLOCK-END
