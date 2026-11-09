"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the unknown function using a diverse basis of polynomial, trigonometric,
    rational, and nonlinear functions. The function is linear in params, which allows BFGS
    to converge quickly since the gradient structure is well-conditioned.
    
    Basis functions (10 total):
    0: 1
    1: x
    2: x^2
    3: x^3
    4: sin(x)
    5: cos(x)
    6: tanh(x)
    7: abs(x)
    8: 1/(1+x^2)
    9: exp(-x^2)
    """
    x = np.asarray(x, dtype=np.float64)
    original_shape = x.shape
    x_flat = x.flatten()
    
    if x_flat.size == 0:
        return np.array([]).reshape(original_shape)
    
    # Compute basis functions
    b0 = np.ones_like(x_flat)
    b1 = x_flat
    b2 = x_flat * x_flat
    b3 = b2 * x_flat
    b4 = np.sin(x_flat)
    b5 = np.cos(x_flat)
    b6 = np.tanh(x_flat)
    b7 = np.abs(x_flat)
    b8 = 1.0 / (1.0 + b2)
    b9 = np.exp(-b2)
    
    # Linear combination
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
    
    # Ensure finite values
    y = np.where(np.isfinite(y), y, 0.0)
    
    return y.reshape(original_shape)
# EVOLVE-BLOCK-END
