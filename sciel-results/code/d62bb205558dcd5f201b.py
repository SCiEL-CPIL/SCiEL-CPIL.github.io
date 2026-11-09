"""Symbolic regression with 1 input using a rich basis of functions."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: y = sum_{i=0}^{9} params[i] * basis_i(x)
    
    Basis functions (10 total):
    0: 1 (constant)
    1: x
    2: x^2
    3: x^3
    4: sin(x)
    5: cos(x)
    6: sin(2x)
    7: cos(2x)
    8: tanh(x)
    9: x * exp(-x^2/4)
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Compute basis functions
    b0 = np.ones_like(x)
    b1 = x
    b2 = x ** 2
    b3 = x ** 3
    b4 = np.sin(x)
    b5 = np.cos(x)
    b6 = np.sin(2.0 * x)
    b7 = np.cos(2.0 * x)
    b8 = np.tanh(x)
    b9 = x * np.exp(-(x ** 2) / 4.0)
    
    # Stack into a matrix (N, 10)
    basis = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    # Linear combination
    y = basis @ params
    
    return y
# EVOLVE-BLOCK-END
