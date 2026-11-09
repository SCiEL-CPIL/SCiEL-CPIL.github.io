import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a linear combination of 10 sinusoidal basis functions.
    The basis functions are fixed, and the parameters are their amplitudes.
    
    Basis:
    0: 1
    1: cos(1.0 * x)
    2: sin(1.0 * x)
    3: cos(2.0 * x)
    4: sin(2.0 * x)
    5: cos(3.0 * x)
    6: sin(3.0 * x)
    7: cos(4.0 * x)
    8: sin(4.0 * x)
    9: cos(5.0 * x)
    """
    # Ensure x is a 1D array for consistent broadcasting
    x = np.asarray(x).flatten()
    
    # Calculate basis functions
    # Using vectorized operations for speed
    b0 = np.ones_like(x)
    b1 = np.cos(1.0 * x)
    b2 = np.sin(1.0 * x)
    b3 = np.cos(2.0 * x)
    b4 = np.sin(2.0 * x)
    b5 = np.cos(3.0 * x)
    b6 = np.sin(3.0 * x)
    b7 = np.cos(4.0 * x)
    b8 = np.sin(4.0 * x)
    b9 = np.cos(5.0 * x)
    
    # Stack basis functions into a matrix (N, 10)
    # Rows correspond to data points, columns to basis functions
    basis_matrix = np.column_stack((b0, b1, b2, b3, b4, b5, b6, b7, b8, b9))
    
    # Compute linear combination
    # params shape (10,), basis_matrix shape (N, 10)
    # Result shape (N,)
    y = basis_matrix @ params
    
    return y
