import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a fixed basis of polynomial, trigonometric, and 
    exponential functions.
    
    Basis:
    0: 1
    1: x
    2: x^2
    3: x^3
    4: sin(x)
    5: cos(x)
    6: sin(2x)
    7: cos(2x)
    8: exp(-x^2)
    9: x^4
    """
    # Ensure x is a 1D array for consistent shape handling
    x = np.asarray(x).flatten()
    
    # Precompute basis functions
    # Using vectorized operations for speed
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = x * x
    b3 = b2 * x
    b4 = np.sin(x)
    b5 = np.cos(x)
    b6 = np.sin(2 * x)
    b7 = np.cos(2 * x)
    b8 = np.exp(-b2)
    b9 = b2 * b2
    
    # Stack into a matrix of shape (N, 10)
    # Using np.column_stack is efficient for this small number of columns
    basis = np.column_stack((b0, b1, b2, b3, b4, b5, b6, b7, b8, b9))
    
    # params is expected to be shape (10,)
    # The result is shape (N,)
    y_pred = basis @ params
    
    return y_pred
