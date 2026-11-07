"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator.
    
    Model:
    a = p0*sin(t) + p1*cos(t) + p2*sin(2t) + p3*cos(2t)  (Multi-harmonic drive)
      + p4 + p5*x + p6*v + p7*x^2 + p8*v^2 + p9*x*v      (Quadratic state dynamics)
    
    This structure allows the optimizer to fit:
    1. A drive with fundamental and second harmonic components.
    2. A local quadratic approximation of the nonlinear restoring and damping forces.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Time-dependent drive terms
    # Using first and second harmonics to capture potential frequency shifts or overtones
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    sin_2t = np.sin(2.0 * t)
    cos_2t = np.cos(2.0 * t)
    
    drive = p0 * sin_t + p1 * cos_t + p2 * sin_2t + p3 * cos_2t
    
    # State-dependent dynamics (Quadratic expansion in x and v)
    x2 = x * x
    v2 = v * v
    xv = x * v
    
    dynamics = p4 + p5 * x + p6 * v + p7 * x2 + p8 * v2 + p9 * xv
    
    return drive + dynamics
# EVOLVE-BLOCK-END
