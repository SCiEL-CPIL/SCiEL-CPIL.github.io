"""Symbolic regression: nonlinear oscillator (2) - closed-form equation."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with time-dependent driving force.
    
    Uses a physics-inspired basis for a driven Duffing-type oscillator:
    - Linear damping and stiffness
    - Cubic nonlinear stiffness (Duffing term)
    - Quadratic damping (aerodynamic-style)
    - Time-dependent forcing via low-order Fourier-like terms in t
    - Cross-coupling terms
    
    10 fitted constants.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize t to a reasonable range for trig functions
    # t is in [30, 50], so center and scale
    t_centered = t - 40.0
    t_scaled = t_centered / 10.0  # roughly [-1, 1]
    
    # Core nonlinear oscillator terms
    # Duffing: x + alpha*x^3
    duffing = x + p0 * x**3
    
    # Damping: linear + quadratic
    damping = p1 * v + p2 * v * np.abs(v)
    
    # Time-dependent driving force (low-order Fourier in t)
    # Using sin and cos of scaled t
    sin_t = np.sin(t_scaled)
    cos_t = np.cos(t_scaled)
    
    # Driving terms with possible modulation by x and v
    drive = (p3 * sin_t + p4 * cos_t + 
             p5 * x * sin_t + p6 * v * cos_t)
    
    # Additional cross terms for flexibility
    cross = p7 * x * v + p8 * t_scaled * x
    
    # Constant offset
    offset = p9
    
    return duffing - damping + drive + cross + offset
# EVOLVE-BLOCK-END
