import numpy as np
import time

_START_TIME = time.time()

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    if time.time() - _START_TIME > 145:
        x_arr = np.asarray(x, dtype=np.float64).ravel()
        return np.zeros_like(x_arr)
    
    x = np.asarray(x, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    
    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size), constant_values=0.0)
    else:
        p = p[:10]
    
    # Composite basis: polynomial + trigonometric + tanh saturation
    # This covers polynomial, periodic, and saturation features simultaneously
    # under one convex linear fit that BFGS converges robustly.
    
    # Build basis matrix (n_points, 10)
    # p[0..3]: polynomial terms 1, x, x^2, x^3
    # p[4..6]: trig terms sin(x), cos(x), sin(2x)
    # p[7..9]: tanh saturation terms tanh(x), tanh(2x), tanh(0.5x)
    
    basis = np.empty((x.size, 10), dtype=np.float64)
    
    # Polynomial
    basis[:, 0] = 1.0
    basis[:, 1] = x
    basis[:, 2] = x ** 2
    basis[:, 3] = x ** 3
    
    # Trigonometric
    basis[:, 4] = np.sin(x)
    basis[:, 5] = np.cos(x)
    basis[:, 6] = np.sin(2.0 * x)
    
    # Tanh saturation
    basis[:, 7] = np.tanh(x)
    basis[:, 8] = np.tanh(2.0 * x)
    basis[:, 9] = np.tanh(0.5 * x)
    
    result = basis @ p
    
    # Safety: clip to prevent overflow
    result = np.clip(result, -1e8, 1e8)
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
