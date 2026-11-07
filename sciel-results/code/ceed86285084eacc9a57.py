"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with a rich basis of 10 functions.
    
    The evaluator fits params (length 10) via BFGS to minimize training MSE.
    We provide a fixed basis of 10 functions that should span common functional
    forms on [-3, 3].
    
    Basis choice:
    - Polynomial terms up to degree 5 (6 terms) to capture smooth trends
    - Sin and cos of x to capture periodic components
    - Exp(-x^2) for Gaussian-like bumps
    - x^2 * exp(-x^2) for skewed Gaussian-like shapes
    
    This gives a flexible linear model that can approximate many common functions.
    """
    # Ensure x is a 1D array
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        x = x.reshape(1)
    
    # Extract the 10 parameters
    w0 = params[0]
    w1 = params[1]
    w2 = params[2]
    w3 = params[3]
    w4 = params[4]
    w5 = params[5]
    w6 = params[6]
    w7 = params[7]
    w8 = params[8]
    w9 = params[9]
    
    # Compute basis functions
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    
    # Polynomial basis: 1, x, x^2, x^3, x^4, x^5
    b0 = np.ones_like(x)
    b1 = x
    b2 = x2
    b3 = x3
    b4 = x4
    b5 = x5
    
    # Trigonometric basis
    b6 = np.sin(x)
    b7 = np.cos(x)
    
    # Exponential/Gaussian basis
    exp_neg_x2 = np.exp(-x2)
    b8 = exp_neg_x2
    b9 = x2 * exp_neg_x2
    
    # Linear combination
    y = (w0 * b0 + w1 * b1 + w2 * b2 + w3 * b3 + w4 * b4 + w5 * b5 +
         w6 * b6 + w7 * b7 + w8 * b8 + w9 * b9)
    
    # Ensure finite output
    y = np.where(np.isfinite(y), y, 0.0)
    
    return y
# EVOLVE-BLOCK-END
