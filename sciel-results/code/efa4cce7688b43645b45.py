"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using Chebyshev polynomials of the first kind.
    
    This implementation uses a separable basis up to degree 3 for each input variable.
    This allows the model to capture stronger non-linearities in individual dimensions
    (up to cubic) which might be more effective than a full quadratic model with
    interactions if the target function is separable or has weak interactions.
    
    Basis functions (10 total):
    1. Constant (T0)
    2-4. Linear terms T1(x), T1(y), T1(z)
    5-7. Quadratic terms T2(x), T2(y), T2(z)
    8-10. Cubic terms T3(x), T3(y), T3(z)
    """
    # Map from [0.5, 2] to [-1, 1]
    # center = 1.25, half_range = 0.75
    center = 1.25
    half_range = 0.75
    
    tx = (x - center) / half_range
    ty = (y - center) / half_range
    tz = (z - center) / half_range
    
    # Do NOT clip, to maintain differentiability for the optimizer.
    # The inputs are guaranteed to be in [0.5, 2], so tx, ty, tz are in [-1, 1].
    
    # Chebyshev polynomials of the first kind
    # T0(t) = 1
    # T1(t) = t
    # T2(t) = 2*t^2 - 1
    # T3(t) = 4*t^3 - 3*t
    
    T0 = np.ones_like(tx)
    
    T1x = tx
    T1y = ty
    T1z = tz
    
    T2x = 2.0 * tx**2 - 1.0
    T2y = 2.0 * ty**2 - 1.0
    T2z = 2.0 * tz**2 - 1.0
    
    T3x = 4.0 * tx**3 - 3.0 * tx
    T3y = 4.0 * ty**3 - 3.0 * ty
    T3z = 4.0 * tz**3 - 3.0 * tz
    
    p = params
    
    result = (
        p[0] * T0 +
        p[1] * T1x +
        p[2] * T1y +
        p[3] * T1z +
        p[4] * T2x +
        p[5] * T2y +
        p[6] * T2z +
        p[7] * T3x +
        p[8] * T3y +
        p[9] * T3z
    )
    
    return result
# EVOLVE-BLOCK-END
