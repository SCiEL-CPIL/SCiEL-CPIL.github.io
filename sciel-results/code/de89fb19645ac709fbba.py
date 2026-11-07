"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of a diverse, smooth basis of 10 functions.
    
    The evaluator fits the 10 params via BFGS starting from all ones,
    minimizing training MSE. We provide a rich but well-conditioned basis
    that can approximate many common hidden functions:
    
    Basis functions:
    0: 1          (constant)
    1: x          (linear)
    2: x^2        (quadratic)
    3: x^3        (cubic)
    4: sin(x)     (periodic)
    5: cos(x)     (periodic)
    6: sin(2x)    (higher frequency periodic)
    7: cos(2x)    (higher frequency periodic)
    8: exp(-x^2)  (Gaussian bump)
    9: tanh(x)    (smooth sigmoid)
    
    This mix covers polynomial, trigonometric, and smooth nonlinear behavior
    with well-conditioned features for BFGS optimization.
    """
    x = np.asarray(x, dtype=np.float64)
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = x * x
    b3 = x * x * x
    b4 = np.sin(x)
    b5 = np.cos(x)
    b6 = np.sin(2.0 * x)
    b7 = np.cos(2.0 * x)
    b8 = np.exp(-x * x)
    b9 = np.tanh(x)
    
    return (params[0] * b0 +
            params[1] * b1 +
            params[2] * b2 +
            params[3] * b3 +
            params[4] * b4 +
            params[5] * b5 +
            params[6] * b6 +
            params[7] * b7 +
            params[8] * b8 +
            params[9] * b9)
# EVOLVE-BLOCK-END
