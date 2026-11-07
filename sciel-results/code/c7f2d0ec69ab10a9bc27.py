"""Symbolic regression with Fourier series expansion."""
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Closed-form prediction using a 10-dim Fourier basis with coefficients from params.
    
    The evaluator fits params via BFGS starting from all ones.
    We use a Fourier series expansion with sine and cosine terms of various
    frequencies to capture periodic and oscillatory target functions.
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
    
    # Build Fourier basis (10 components to match params length)
    b = np.empty((10, n), dtype=np.float64)
    
    # Constant term
    b[0] = 1.0
    
    # Sine and cosine of x
    b[1] = np.sin(x)
    b[2] = np.cos(x)
    
    # Sine and cosine of y
    b[3] = np.sin(y)
    b[4] = np.cos(y)
    
    # Sine and cosine of x+y
    b[5] = np.sin(x + y)
    b[6] = np.cos(x + y)
    
    # Sine and cosine of x-y
    b[7] = np.sin(x - y)
    b[8] = np.cos(x - y)
    
    # Higher frequency: sin(2x)
    b[9] = np.sin(2.0 * x)
    
    # Linear combination
    prediction = params @ b
    
    # Ensure finite
    if not np.all(np.isfinite(prediction)):
        prediction = np.where(np.isfinite(prediction), prediction, 0.0)
    
    return prediction
