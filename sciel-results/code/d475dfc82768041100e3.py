"""Symbolic regression: equation for a nonlinear oscillator."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form acceleration for a nonlinear oscillator.
    
    Uses a complete cubic polynomial basis in normalized coordinates.
    Normalization improves the conditioning of the Jacobian for BFGS.
    
    Basis terms (10 total):
    1. 1
    2. x_s
    3. v_s
    4. x_s^2
    5. x_s*v_s
    6. v_s^2
    7. x_s^3
    8. x_s^2*v_s
    9. x_s*v_s^2
    10. v_s^3
    
    where x_s = x / 0.57, v_s = v / 0.28
    """
    # Normalize inputs to O(1) scale based on training range max abs values
    # x range: [-0.5728, 0.5275] -> max abs ~ 0.57
    # v range: [-0.1754, 0.2821] -> max abs ~ 0.28
    x_s = x / 0.57
    v_s = v / 0.28
    
    # Compute powers
    x_s2 = x_s * x_s
    x_s3 = x_s2 * x_s
    v_s2 = v_s * v_s
    v_s3 = v_s2 * v_s
    xv_s = x_s * v_s
    x2v_s = x_s2 * v_s
    xv2_s = x_s * v_s2
    
    # Basis vectors
    b0 = np.ones_like(x_s)
    b1 = x_s
    b2 = v_s
    b3 = x_s2
    b4 = xv_s
    b5 = v_s2
    b6 = x_s3
    b7 = x2v_s
    b8 = xv2_s
    b9 = v_s3
    
    # Linear combination of basis functions
    return (params[0] * b0 + params[1] * b1 + params[2] * b2 +
            params[3] * b3 + params[4] * b4 + params[5] * b5 +
            params[6] * b6 + params[7] * b7 + params[8] * b8 +
            params[9] * b9)
# EVOLVE-BLOCK-END
