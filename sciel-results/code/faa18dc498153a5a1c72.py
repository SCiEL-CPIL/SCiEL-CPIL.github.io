"""Cubic polynomial model for nonlinear oscillator acceleration."""
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a full cubic polynomial basis.
    
    Parameters:
        x: position array
        v: velocity array
        params: array of 10 values
            p[0] = bias (1)
            p[1] = x
            p[2] = v
            p[3] = x^2
            p[4] = x*v
            p[5] = v^2
            p[6] = x^3
            p[7] = x^2*v
            p[8] = x*v^2
            p[9] = v^3
    """
    # Compute basis functions
    # Note: x and v are expected to be 1D arrays of same length
    
    # Degree 0
    b0 = np.ones_like(x)
    
    # Degree 1
    b1 = x
    b2 = v
    
    # Degree 2
    b3 = x * x
    b4 = x * v
    b5 = v * v
    
    # Degree 3
    b6 = x * x * x
    b7 = x * x * v
    b8 = x * v * v
    b9 = v * v * v
    
    # Linear combination
    # Using einsum or dot product is efficient, but explicit sum is clear and fast enough for 10000 rows
    # y = p0*b0 + p1*b1 + ... + p9*b9
    
    y = (params[0] * b0 +
         params[1] * b1 +
         params[2] * b2 +
         params[3] * b3 +
         params[4] * b4 +
         params[5] * b5 +
         params[6] * b6 +
         params[7] * b7 +
         params[8] * b8 +
         params[9] * b9)
         
    return y
