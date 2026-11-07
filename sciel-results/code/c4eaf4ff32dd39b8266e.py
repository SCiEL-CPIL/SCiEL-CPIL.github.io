"""Symbolic regression with a mixed Fourier-polynomial basis."""
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Closed-form prediction using a 10-dim mixed basis.
    
    The evaluator fits params via BFGS starting from all ones.
    We use a mixed basis combining low-order polynomial terms with
    sine/cosine terms to capture both smooth non-periodic and periodic
    structure in the target function.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Ensure correct shape
    if x.ndim != 1:
        x = x.reshape(-1)
    if y.ndim != 1:
        y = y.reshape(-1)
    
    n = x.size
    
    # Build mixed basis (10 components to match params length)
    b = np.empty((10, n), dtype=np.float64)
    
    # Constant
    b[0] = 1.0
    
    # Linear terms
    b[1] = x
    b[2] = y
    
    # Quadratic terms
    b[3] = x * x
    b[4] = x * y
    b[5] = y * y
    
    # Sine and cosine of x
    b[6] = np.sin(x)
    b[7] = np.cos(x)
    
    # Sine and cosine of y
    b[8] = np.sin(y)
    b[9] = np.cos(y)
    
    # Linear combination
    prediction = params @ b
    
    # Ensure finite
    if not np.all(np.isfinite(prediction)):
        prediction = np.where(np.isfinite(prediction), prediction, 0.0)
    
    return prediction
