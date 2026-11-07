# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 1 input.
    Uses a combination of polynomial and trigonometric basis functions
    to approximate the hidden function.
    
    params: array of length 10
      p0, p1: linear coefficients (1, x)
      p2, p3, p4: quadratic/cubic coefficients (x^2, x^3, x^4)
      p5, p6: sin, cos coefficients (sin(x), cos(x))
      p7, p8: sin(2x), cos(2x) coefficients
      p9: tanh(x) coefficient
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Ensure params is a flat array of length 10
    p = np.asarray(params, dtype=np.float64).ravel()
    if p.shape[0] < 10:
        # Pad with zeros if necessary (should not happen in normal use)
        p = np.pad(p, (0, 10 - p.shape[0]))
    else:
        p = p[:10]
    
    # Basis functions
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_2x = np.sin(2.0 * x)
    cos_2x = np.cos(2.0 * x)
    tanh_x = np.tanh(x)
    
    # Combine: 
    # p0 * 1 + p1 * x + p2 * x^2 + p3 * x^3 + p4 * x^4 
    # + p5 * sin(x) + p6 * cos(x) + p7 * sin(2x) + p8 * cos(2x) + p9 * tanh(x)
    
    result = (
        p[0] 
        + p[1] * x 
        + p[2] * x2 
        + p[3] * x3 
        + p[4] * x4 
        + p[5] * sin_x 
        + p[6] * cos_x 
        + p[7] * sin_2x 
        + p[8] * cos_2x 
        + p[9] * tanh_x
    )
    
    return result
# EVOLVE-BLOCK-END
