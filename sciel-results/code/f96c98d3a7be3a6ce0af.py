"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model for a nonlinear oscillator acceleration.
    
    Physics-inspired terms:
    - Duffing stiffness: x^3 (nonlinear spring)
    - Linear stiffness: x
    - Van der Pol damping: (1 - x^2) * v (self-excited oscillation)
    - Linear damping: v
    - Periodic drive: sin(t), cos(t)
    - Quadratic nonlinearity: x^2
    - Cross term: x * v
    - Constant offset
    """
    p = params
    # Ensure we have at least 10 params (pad with zeros if needed)
    if len(p) < 10:
        p = np.concatenate([p, np.zeros(10 - len(p))])
    
    # Nonlinear oscillator terms
    x2 = x * x
    x3 = x2 * x
    
    # Core terms
    result = (
        p[0] * x3          # Duffing cubic stiffness
        + p[1] * x         # Linear stiffness
        + p[2] * v         # Linear damping
        + p[3] * x2 * v    # Van der Pol nonlinear damping (x^2 * v)
        + p[4] * np.sin(t) # Periodic drive (sin)
        + p[5] * np.cos(t) # Periodic drive (cos)
        + p[6] * x2        # Quadratic nonlinearity
        + p[7] * x * v     # Cross term
        + p[8] * v * v     # Velocity squared (drag-like)
        + p[9]             # Constant offset
    )
    
    return result
# EVOLVE-BLOCK-END
