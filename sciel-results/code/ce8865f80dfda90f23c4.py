"""Symbolic regression with rich standardized basis for BFGS fitting."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rich basis function with 10 parameters.
    
    The design uses a diverse set of basis functions to cover common functional
    forms: linear, quadratic, cubic, exponential, logarithmic, power, trigonometric,
    and rational functions. The parameters are fitted by BFGS starting from ones.
    
    To improve conditioning, we use a mix of well-scaled features.
    
    params[0:10] are coefficients for 10 basis functions:
    0: 1 (bias)
    1: x
    2: x^2
    3: x^3
    4: exp(x)
    5: exp(-x)
    6: log(x)
    7: 1/x
    8: sin(x)
    9: cos(x)
    """
    # Ensure x is a 1D array
    x = np.asarray(x, dtype=np.float64)
    
    # Clip x to avoid issues with log and 1/x near zero (x is in [0.1, 6])
    x_safe = np.clip(x, 1e-10, None)
    
    # Build basis
    b0 = np.ones_like(x_safe)
    b1 = x_safe
    b2 = x_safe ** 2
    b3 = x_safe ** 3
    b4 = np.exp(np.clip(x_safe, -50, 50))
    b5 = np.exp(-x_safe)
    b6 = np.log(x_safe)
    b7 = 1.0 / x_safe
    b8 = np.sin(x_safe)
    b9 = np.cos(x_safe)
    
    # Stack into matrix: shape (n, 10)
    basis = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    # params is shape (10,)
    params = np.asarray(params, dtype=np.float64).reshape(-1)
    
    # Predict: (n, 10) @ (10,) -> (n,)
    pred = basis @ params
    
    return pred
# EVOLVE-BLOCK-END
