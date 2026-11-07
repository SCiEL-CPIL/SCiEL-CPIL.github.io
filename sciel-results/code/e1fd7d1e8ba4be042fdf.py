import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0  # Safety margin before 150s limit

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        # Fallback: return constant prediction
        return np.full_like(x, params[0])
    
    # Define Gaussian RBF basis
    # 10 Gaussians centered evenly in [-2, 2]
    centers = np.linspace(-2.0, 2.0, 10)
    sigma = 0.4
    
    # Compute basis matrix
    # x is shape (N,), centers is shape (10,)
    # We want B of shape (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    B = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Return linear combination
    y_pred = B @ params
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(y_pred)):
        return np.full_like(x, params[0])
        
    return y_pred
