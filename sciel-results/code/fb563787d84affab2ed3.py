import numpy as np
import time

# Global timer for deadline guard
_start_time = time.time()
DEADLINE = 110.0  # 10s margin before 120s limit

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers hidden function using a Gaussian Radial Basis Function (RBF) model.
    This is a structural change from polynomial basis to local kernel basis.
    The model is: y = sum_{i=0}^{9} p_i * exp(-(x - c_i)^2 / (2 * sigma^2))
    This allows for better approximation of non-polynomial features like
    sharp transitions or localized bumps.
    """
    # Deadline guard
    if time.time() - _start_time > DEADLINE:
        return np.zeros_like(x, dtype=np.float64)

    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Define 10 Gaussian centers spread across [-3, 3]
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Fixed width (sigma) for all Gaussians
    # A smaller sigma makes the basis more local; larger makes it more global.
    # sigma = 1.0 is a reasonable default for [-3, 3].
    sigma = 1.0
    
    # Compute the basis matrix T where T[j, i] = exp(-(x[j] - centers[i])^2 / (2 * sigma^2))
    # x shape: (N,), centers shape: (10,)
    # We want T shape: (N, 10)
    
    # Efficient computation:
    # diff = x[:, None] - centers[None, :]
    # T = np.exp(-diff**2 / (2 * sigma**2))
    
    # Handling edge cases for memory if x is very large, but N=200 is small.
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    T = np.exp(-diff**2 / (2.0 * sigma**2))
    
    # Linear combination
    y = T @ p
    
    # Ensure finiteness
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        
    return y
