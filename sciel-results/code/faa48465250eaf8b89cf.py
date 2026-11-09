"""Symbolic regression with a rich, numerically stable basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to roughly [-1, 1] for numerical stability
    # Inputs are in [0.5, 2.0], center=1.25, half_range=0.75
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75

    p = params

    # Build a 10-feature basis:
    # 0: constant
    # 1: u
    # 2: v
    # 3: w
    # 4: u*v
    # 5: v*w
    # 6: w*u
    # 7: u^2
    # 8: v^2
    # 9: w^2
    # This captures linear, pairwise multiplicative, and quadratic terms.
    # It's linear in params, so BFGS will converge reliably (convex problem).
    
    return (p[0] 
            + p[1] * u 
            + p[2] * v 
            + p[3] * w 
            + p[4] * u * v 
            + p[5] * v * w 
            + p[6] * w * u 
            + p[7] * u * u 
            + p[8] * v * v 
            + p[9] * w * w)
# EVOLVE-BLOCK-END
