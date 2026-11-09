"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear damped oscillator with periodic driving.
    
    Model: a = A*sin(w*t + phi) + c1*x + c3*x^3 + d1*v + d3*v^3 + c12*x*v + c2*x^2 + c2v*x*v^2 + bias
    This captures:
    - Periodic driving force (sin with fitted amplitude, frequency, phase)
    - Linear and cubic restoring force (Duffing-like)
    - Linear and cubic damping
    - Cross-coupling terms
    """
    A = params[0]       # drive amplitude
    w = params[1]       # drive frequency
    phi = params[2]     # drive phase
    c1 = params[3]      # linear stiffness
    c3 = params[4]      # cubic stiffness (Duffing)
    d1 = params[5]      # linear damping
    d3 = params[6]      # cubic damping
    c12 = params[7]     # x*v coupling
    c2 = params[8]      # quadratic position
    bias = params[9]    # constant bias
    
    # Periodic driving
    drive = A * np.sin(w * t + phi)
    
    # Restoring forces
    restoring = c1 * x + c3 * x**3 + c2 * x**2
    
    # Damping forces
    damping = d1 * v + d3 * v**3
    
    # Coupling
    coupling = c12 * x * v
    
    return drive + restoring + damping + coupling + bias
# EVOLVE-BLOCK-END
