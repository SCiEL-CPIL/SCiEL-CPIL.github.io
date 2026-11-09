"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 150.0
_TIME_MARGIN = 5.0  # safety margin in seconds


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Normalized Full Quadratic Model.
    
    Structural Change:
    1. Input Normalization: Map inputs from [0.5, 2] to [-1, 1] to improve
       the conditioning of the design matrix for BFGS. This helps the optimizer
       avoid getting stuck due to scale differences between terms like x and x^2.
    2. Basis Selection: Use a full quadratic basis (1, x, y, z, x^2, y^2, z^2, xy, xz, yz).
       This covers a wide range of smooth functions. If the target is a polynomial
       of degree <= 2, this model is exact. If not, it provides the best quadratic
       approximation, which is often better than sparse non-linear bases when
       the optimizer is constrained.
    
    Basis allocation (10 params):
    p0: bias
    p1: x_n
    p2: y_n
    p3: z_n
    p4: x_n^2
    p5: y_n^2
    p6: z_n^2
    p7: x_n*y_n
    p8: x_n*z_n
    p9: y_n*z_n
    """
    # Check time budget
    if time.time() - _START_TIME > _TIME_LIMIT - _TIME_MARGIN:
        # Fallback: simple mean prediction (params[0] is bias, others ignored effectively)
        # Or just return params[0] as a safe constant
        return np.full_like(x, params[0], dtype=float)
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    
    # Normalize inputs: Center at 1.25 (midpoint of 0.5, 2), Scale by 0.75 (half-range)
    # x_n = (x - 1.25) / 0.75
    # This maps [0.5, 2] to [-1, 1]
    x_n = (x - 1.25) / 0.75
    y_n = (y - 1.25) / 0.75
    z_n = (z - 1.25) / 0.75
    
    p = params
    
    # Build the prediction using normalized inputs
    pred = (p[0] * np.ones_like(x) +
            p[1] * x_n +
            p[2] * y_n +
            p[3] * z_n +
            p[4] * x_n**2 +
            p[5] * y_n**2 +
            p[6] * z_n**2 +
            p[7] * x_n * y_n +
            p[8] * x_n * z_n +
            p[9] * y_n * z_n)
    
    return pred
# EVOLVE-BLOCK-END
