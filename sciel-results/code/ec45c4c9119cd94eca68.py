import numpy as np
import time

N = 400

def _compute_objective(x):
    """Compute max_k |autocorr[k]| / (sum(x))^2."""
    S = np.sum(x)
    if abs(S) < 1e-12:
        return np.inf
    # Compute autoconvolution
    autocorr = np.correlate(x, x, mode='full')
    # autocorr has length 2*N-1, centered at index N-1
    # We want max absolute value
    max_abs = np.max(np.abs(autocorr))
    return max_abs / (S * S)

def construct():
    deadline = time.time() + 140.0  # Leave 10s margin
    
    # Start with all ones (known good baseline)
    x = np.ones(N)
    best_obj = _compute_objective(x)
    best_x = x.copy()
    
    # Local search: flip random positions
    np.random.seed(42)
    for i in range(100000):
        if time.time() > deadline:
            break
        
        # Flip a random position
        idx = np.random.randint(N)
        x[idx] = -x[idx]
        
        obj = _compute_objective(x)
        if obj < best_obj:
            best_obj = obj
            best_x = x.copy()
        else:
            x[idx] = -x[idx]  # Revert
    
    return best_x
