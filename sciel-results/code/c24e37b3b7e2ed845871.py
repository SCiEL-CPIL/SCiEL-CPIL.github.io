import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid trigonometric-polynomial basis.
    Basis:
    0: 1
    1: x
    2: cos(x)
    3: sin(x)
    4: cos(2x)
    5: sin(2x)
    6: cos(3x)
    7: sin(3x)
    8: cos(4x)
    9: sin(4x)
    """
    # Ensure x is a 1D array
    x = np.asarray(x, dtype=float).ravel()
    
    # Build the basis matrix
    # We can do this vectorized for speed, though N=200 is small.
    # B has shape (N, 10)
    
    # Precompute trig functions
    cos_x = np.cos(x)
    sin_x = np.sin(x)
    cos_2x = np.cos(2 * x)
    sin_2x = np.sin(2 * x)
    cos_3x = np.cos(3 * x)
    sin_3x = np.sin(3 * x)
    cos_4x = np.cos(4 * x)
    sin_4x = np.sin(4 * x)
    
    # Stack columns
    # Note: np.column_stack is efficient enough for N=200
    B = np.column_stack((
        np.ones_like(x),
        x,
        cos_x,
        sin_x,
        cos_2x,
        sin_2x,
        cos_3x,
        sin_3x,
        cos_4x,
        sin_4x
    ))
    
    # Prediction
    return B @ params
