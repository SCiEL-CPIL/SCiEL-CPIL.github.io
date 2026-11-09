"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression model with 10 parameters.
    
    The basis functions are designed to be stable for BFGS optimization
    starting from all ones. We use a combination of:
    - Polynomial terms (up to degree 4)
    - Common non-linear activations (tanh, sigmoid)
    - Gaussian-like terms
    - Bounded terms to avoid overflow
    
    All basis functions are bounded or grow slowly on [-2, 2].
    """
    p = params
    
    # Clip x to avoid extreme values (though inputs are in [-2, 2])
    x_safe = np.clip(x, -10, 10)
    
    # Precompute powers
    x2 = x_safe * x_safe
    x3 = x2 * x_safe
    x4 = x2 * x2
    
    # Non-linear activations
    tanh_x = np.tanh(x_safe)
    sig_x = 1.0 / (1.0 + np.exp(-np.clip(x_safe, -20, 20)))
    
    # Gaussian terms
    exp_neg_x2 = np.exp(-x2)
    exp_neg_2x2 = np.exp(-2.0 * x2)
    
    # Basis functions - all bounded or slow-growing on [-2, 2]
    b0 = x_safe          # linear
    b1 = np.ones_like(x_safe)  # constant
    b2 = x2              # quadratic
    b3 = x3              # cubic
    b4 = x4              # quartic
    b5 = tanh_x          # tanh
    b6 = sig_x           # sigmoid
    b7 = x_safe * tanh_x # x * tanh(x)
    b8 = exp_neg_x2      # exp(-x^2)
    b9 = x_safe * exp_neg_2x2  # x * exp(-2x^2)
    
    # Combine with parameters
    result = (
        p[0] * b0 +
        p[1] * b1 +
        p[2] * b2 +
        p[3] * b3 +
        p[4] * b4 +
        p[5] * b5 +
        p[6] * b6 +
        p[7] * b7 +
        p[8] * b8 +
        p[9] * b9
    )
    
    # Ensure finite output
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return result
# EVOLVE-BLOCK-END
