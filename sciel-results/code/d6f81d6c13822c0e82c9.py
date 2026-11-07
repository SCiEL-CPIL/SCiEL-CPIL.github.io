# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression with 10 parameters for symbolic regression.
    
    Uses a rich basis combining polynomial, rational, and reciprocal terms
    to span a large function space including asymptotic/hyperbolic behavior.
    
    Basis functions (10 total):
    0: x
    1: y
    2: z
    3: x*y
    4: x*z
    5: y*z
    6: 1/x
    7: 1/y
    8: 1/z
    9: x*y*z
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure no division by zero (inputs are in [0.5, 2] so safe, but be robust)
    x_safe = np.where(np.abs(x) < 1e-12, 1e-12, x)
    y_safe = np.where(np.abs(y) < 1e-12, 1e-12, y)
    z_safe = np.where(np.abs(z) < 1e-12, 1e-12, z)
    
    pred = (
        p0 * x +
        p1 * y +
        p2 * z +
        p3 * x * y +
        p4 * x * z +
        p5 * y * z +
        p6 / x_safe +
        p7 / y_safe +
        p8 / z_safe +
        p9 * x * y * z
    )
    
    return pred
# EVOLVE-BLOCK-END
