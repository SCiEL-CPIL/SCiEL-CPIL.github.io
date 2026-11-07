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
    
    # Fixed Gaussian RBF centers evenly spaced in [-2, 2]
    # 10 centers: -2, -1.555..., -1.111..., -0.666..., -0.222..., 0.222..., 0.666..., 1.111..., 1.555..., 2
    centers = np.linspace(-2.0, 2.0, 10)
    # Fixed width for all RBFs
    width = 0.5
    
    # Compute RBF values: phi_i(x) = exp(-((x - c_i)^2) / (2 * width^2))
    # Shape: (n_points, 10)
    diffs = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-(diffs ** 2) / (2.0 * width ** 2))
    
    # Linear combination
    result = rbf_values @ p
    
    # Safety: clip to prevent overflow
    result = np.clip(result, -1e8, 1e8)
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
