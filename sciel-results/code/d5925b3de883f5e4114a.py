"""Symbolic regression: nonlinear oscillator acceleration from (x, v)."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator modeled as a cubic polynomial in
    scaled position and velocity.
    
    Uses 10 parameters for a full cubic basis in scaled (x, v):
    p0: 1
    p1: X
    p2: X^2
    p3: X^3
    p4: V
    p5: V^2
    p6: V^3
    p7: X*V
    p8: X^2*V
    p9: X*V^2
    
    where X = x/scale_x, V = v/scale_v.
    """
    # Scale inputs based on training ranges for numerical conditioning
    # x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]
    sx = 0.55
    sv = 0.25
    
    X = x / sx
    V = v / sv
    
    X2 = X * X
    X3 = X2 * X
    V2 = V * V
    V3 = V2 * V
    XV = X * V
    X2V = X2 * V
    XV2 = X * V2
    
    return (params[0] +
            params[1] * X +
            params[2] * X2 +
            params[3] * X3 +
            params[4] * V +
            params[5] * V2 +
            params[6] * V3 +
            params[7] * XV +
            params[8] * X2V +
            params[9] * XV2)
# EVOLVE-BLOCK-END
