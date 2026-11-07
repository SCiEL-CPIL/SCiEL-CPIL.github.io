"""Symbolic regression: nonlinear oscillator acceleration from position and velocity."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Cubic Polynomial Basis model for nonlinear oscillator acceleration.
    
    Uses a full cubic polynomial in x and v.
    Basis functions: 1, x, v, x^2, xv, v^2, x^3, x^2v, xv^2, v^3
    
    Total 10 coefficients, matching the params array length.
    
    Prediction: y = sum_i params[i] * basis_i(x, v)
    """
    # Ensure x and v are 1D arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    
    # Compute basis functions
    # Order: 1, x, v, x^2, xv, v^2, x^3, x^2v, xv^2, v^3
    b0 = np.ones_like(x)
    b1 = x
    b2 = v
    b3 = x * x
    b4 = x * v
    b5 = v * v
    b6 = x * x * x
    b7 = x * x * v
    b8 = x * v * v
    b9 = v * v * v
    
    # Combine with parameters
    # params is expected to be length 10
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    prediction = (p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3 + p4 * b4 +
                  p5 * b5 + p6 * b6 + p7 * b7 + p8 * b8 + p9 * b9)
    
    return prediction
# EVOLVE-BLOCK-END
