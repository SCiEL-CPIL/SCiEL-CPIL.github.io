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
    
    # Use Radial Basis Functions (Gaussians) with a wider spread.
    # Previous RBF width 0.5 scored 939. Inspiration suggests 0.5 is too narrow.
    # Wider RBFs (width 1.0-1.5) provide smoother interpolation and better generalization
    # for functions that are not extremely spiky.
    
    centers = np.linspace(-2.0, 2.0, 10)
    width = 1.2
    
    # Compute RBF values: phi_i(x) = exp(-((x - c_i)^2) / (2 * width^2))
    diffs = x[:, np.newaxis] - centers[np.newaxis, :]
    basis = np.exp(-(diffs ** 2) / (2.0 * width ** 2))
    
    result = basis @ p
    
    # Safety: clip to prevent overflow
    result = np.clip(result, -1e8, 1e8)
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
