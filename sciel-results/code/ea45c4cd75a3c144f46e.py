"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model for a nonlinear oscillator acceleration.
    
    Physics-inspired terms for a driven, damped nonlinear oscillator:
    - Linear and cubic stiffness (Duffing-type)
    - Linear and nonlinear damping (Van der Pol-type)
    - Periodic driving force
    - Cross-coupling terms
    """
    p = params
    # Ensure we have at least 10 params (pad with zeros if needed)
    if len(p) < 10:
        p = np.concatenate([p, np.zeros(10 - len(p))])
    
    # Precompute powers and products
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    
    # Core oscillator terms
    # p0: linear stiffness coefficient
    # p1: cubic (Duffing) stiffness coefficient  
    # p2: linear damping coefficient
    # p3: nonlinear damping (x^2 * v) coefficient
    # p4: quadratic position term
    # p5: velocity squared (drag) term
    # p6: cross term x*v
    # p7: sin drive amplitude
    # p8: cos drive amplitude
    # p9: constant offset
    
    result = (
        p[0] * x +          # Linear restoring force
        p[1] * x3 +         # Cubic (Duffing) restoring force
        p[2] * v +          # Linear damping
        p[3] * x2 * v +     # Nonlinear (Van der Pol) damping
        p[4] * x2 +         # Quadratic position correction
        p[5] * v2 +         # Velocity squared (quadratic drag)
        p[6] * x * v +      # Cross-coupling term
        p[7] * np.sin(t) +  # Sinusoidal drive
        p[8] * np.cos(t) +  # Cosinusoidal drive
        p[9]                # Constant offset
    )
    
    return result
# EVOLVE-BLOCK-END
