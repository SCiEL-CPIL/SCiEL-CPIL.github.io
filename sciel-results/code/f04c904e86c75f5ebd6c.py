"""Nonlinear oscillator acceleration model with driven Duffing dynamics."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a driven, damped Duffing oscillator.
    
    Model:
    a = A*sin(w*t + phi) + c*x + d*x^3 + e*v + f*v^3 + g*x*v + h*x^2 + k*v^2 + b
    
    Parameters (10 total):
    p0: drive amplitude A
    p1: drive frequency w
    p2: drive phase phi
    p3: linear stiffness c
    p4: cubic stiffness d (Duffing term)
    p5: linear damping e
    p6: cubic damping f
    p7: position-velocity coupling g
    p8: quadratic position h
    p9: constant bias b
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Sinusoidal driving force
    drive = p0 * np.sin(p1 * t + p2)
    
    # Restoring forces (linear + Duffing + quadratic)
    restoring = p3 * x + p4 * x**3 + p8 * x**2
    
    # Damping forces (linear + cubic)
    damping = p5 * v + p6 * v**3
    
    # Cross-coupling term
    coupling = p7 * x * v
    
    # Constant bias
    bias = p9
    
    return drive + restoring + damping + coupling + bias
# EVOLVE-BLOCK-END
