"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator.
    
    Uses a complete cubic polynomial basis in scaled inputs x and v.
    Scaling inputs to [-1, 1] improves the condition number of the 
    design matrix, aiding the BFGS optimizer in finding the global 
    minimum for the coefficients.
    """
    p = params
    
    # Scale inputs based on training ranges to improve conditioning
    # x range: [-0.5728, 0.5275] -> max abs approx 0.5728
    # v range: [-0.1754, 0.2821] -> max abs approx 0.2821
    x_s = x / 0.5728
    v_s = v / 0.2821
    
    # Pre-compute monomials
    x2 = x_s * x_s
    x3 = x2 * x_s
    v2 = v_s * v_s
    v3 = v2 * v_s
    xv = x_s * v_s
    x2v = x2 * v_s
    xv2 = x_s * v2
    
    # Linear combination of the 10 basis functions
    # Basis: [1, x_s, v_s, x_s^2, x_s^3, v_s^2, x_s*v_s, x_s^2*v_s, x_s*v_s^2, v_s^3]
    return (p[0] + 
            p[1] * x_s + 
            p[2] * v_s + 
            p[3] * x2 + 
            p[4] * x3 + 
            p[5] * v2 + 
            p[6] * xv + 
            p[7] * x2v + 
            p[8] * xv2 + 
            p[9] * v3)
# EVOLVE-BLOCK-END
