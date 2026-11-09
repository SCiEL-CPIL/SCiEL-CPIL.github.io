import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Tensor-product Fourier basis with fundamental frequency pi (period 2)
    # on [-1, 1] domain. This aligns with the domain and provides an orthogonal
    # basis for stable BFGS optimization.
    
    c0 = np.ones_like(x)
    
    # x-basis: cos(pi*x), sin(pi*x), cos(2*pi*x), sin(2*pi*x)
    cx1 = np.cos(np.pi * x)
    sx1 = np.sin(np.pi * x)
    cx2 = np.cos(2.0 * np.pi * x)
    sx2 = np.sin(2.0 * np.pi * x)
    
    # y-basis: cos(pi*y), sin(pi*y)
    cy1 = np.cos(np.pi * y)
    sy1 = np.sin(np.pi * y)
    
    # Construct prediction using 10 parameters
    # Terms: constant, x-only (2), y-only (2), xy-interactions (4), linear residual (1)
    pred = (params[0] * c0 +
            params[1] * cx1 +
            params[2] * sx1 +
            params[3] * cy1 +
            params[4] * sy1 +
            params[5] * cx1 * cy1 +
            params[6] * cx1 * sy1 +
            params[7] * sx1 * cy1 +
            params[8] * sx1 * sy1 +
            params[9] * (x + y))
            
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = params[0] + params[1] * x + params[2] * y
        if not np.all(np.isfinite(pred)):
            pred = np.zeros_like(x)
            
    return pred
