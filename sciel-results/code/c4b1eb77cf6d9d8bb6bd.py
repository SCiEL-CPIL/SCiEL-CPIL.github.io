import numpy as np
import time

# Hard deadline guard: return safely before the 150 s limit.
_START = time.time()
_DEADLINE = _START + 140.0

def _rbf_basis(x: np.ndarray) -> np.ndarray:
    """
    Build a Gaussian RBF basis with 10 fixed, evenly-spaced centers across [0.1, 6].
    Each basis function is exp(-((x - c_j) / sigma)^2) with a fixed width sigma.
    The local nature decouples parameter interactions, giving BFGS a well-conditioned
    landscape where each param acts as an independent local weight.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    
    x_min = 0.1
    x_max = 6.0
    n_centers = 10
    
    # Evenly spaced centers covering the domain
    centers = np.linspace(x_min, x_max, n_centers)
    
    # Width: ~1/9 of the domain span so neighbors overlap for smoothness
    sigma = (x_max - x_min) / 9.0
    
    N = x.size
    B = np.empty((N, n_centers), dtype=np.float64)
    
    # Precompute 1/sigma^2
    inv_sigma2 = 1.0 / (sigma * sigma)
    
    for j in range(n_centers):
        diff = x - centers[j]
        B[:, j] = np.exp(-inv_sigma2 * diff * diff)
    
    return B

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form predictor using Gaussian RBF basis with fixed centers/widths.
    Linear in parameters, so BFGS converges quickly to the global minimum.
    Local basis decouples parameters for a well-conditioned landscape.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    x = x.astype(np.float64)
    
    if x.size == 0:
        return np.zeros(0, dtype=np.float64)
    
    p = np.asarray(params, dtype=np.float64)
    if p.ndim != 1 or p.shape[0] != 10:
        return np.zeros_like(x)
    
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.zeros_like(x)
    
    B = _rbf_basis(x)
    
    out = B @ p
    if not np.all(np.isfinite(out)):
        return np.zeros_like(x)
    return out
