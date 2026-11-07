"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a linear combination of fixed basis functions.
    The model is linear in params, making BFGS convergence reliable.
    
    Parameters (length 10):
    p0: constant
    p1: x
    p2: x^2
    p3: x^3
    p4: x^4
    p5: sin(x)
    p6: cos(x)
    p7: tanh(x)
    p8: exp(x)
    p9: exp(-x)
    """
    # Build basis matrix: each row is a basis function evaluated at x
    # x shape: (n,)
    # basis shape: (n, 10)
    
    # Clip x to avoid extreme values in exp if x were larger, though x in [-2,2]
    x_safe = np.clip(x, -50, 50)
    
    b0 = np.ones_like(x_safe)
    b1 = x_safe
    b2 = x_safe ** 2
    b3 = x_safe ** 3
    b4 = x_safe ** 4
    b5 = np.sin(x_safe)
    b6 = np.cos(x_safe)
    b7 = np.tanh(x_safe)
    b8 = np.exp(x_safe)
    b9 = np.exp(-x_safe)
    
    # Stack basis functions into a matrix (n, 10)
    # Using np.column_stack or np.vstack with axis
    basis = np.column_stack((b0, b1, b2, b3, b4, b5, b6, b7, b8, b9))
    
    # params shape: (10,)
    # prediction = basis @ params
    y = basis @ params
    
    return y
# EVOLVE-BLOCK-END
