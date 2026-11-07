"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier Series Model: Linear + Sine/Cosine harmonics.
    
    This approach uses a basis of trigonometric functions which is linear in the
    parameters. This ensures that the optimizer (BFGS) will converge to the
    global minimum for this model class, avoiding local minima issues associated
    with non-linear parameter dependencies (like Gaussian centers/widths).
    
    Params:
    p0: Intercept
    p1: Linear coefficient
    p2, p3: Amplitude for sin(x), cos(x)
    p4, p5: Amplitude for sin(2x), cos(2x)
    p6, p7: Amplitude for sin(3x), cos(3x)
    p8, p9: Amplitude for sin(4x), cos(4x)
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Extract parameters
    p0, p1 = params[0], params[1]
    a1, b1 = params[2], params[3]
    a2, b2 = params[4], params[5]
    a3, b3 = params[6], params[7]
    a4, b4 = params[8], params[9]
    
    # Compute basis functions
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_2x = np.sin(2 * x)
    cos_2x = np.cos(2 * x)
    sin_3x = np.sin(3 * x)
    cos_3x = np.cos(3 * x)
    sin_4x = np.sin(4 * x)
    cos_4x = np.cos(4 * x)
    
    # Linear combination
    y = p0 + p1 * x
    y += a1 * sin_x + b1 * cos_x
    y += a2 * sin_2x + b2 * cos_2x
    y += a3 * sin_3x + b3 * cos_3x
    y += a4 * sin_4x + b4 * cos_4x
    
    return y
# EVOLVE-BLOCK-END
