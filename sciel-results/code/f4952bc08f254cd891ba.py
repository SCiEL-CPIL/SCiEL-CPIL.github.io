"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START
def _objective(params):
    """Compute the value for a given two-level pattern."""
    p = params[0]  # fraction of +1 positions
    a = params[1]  # positive height
    b = params[2]  # negative height
    
    # Build pattern: first N*p positions get +a, rest get +b
    n_pos = int(round(p * N))
    n_pos = max(1, min(N - 1, n_pos))
    
    h = np.zeros(N)
    h[:n_pos] = a
    h[n_pos:] = b
    
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    
    conv = np.convolve(h, h)
    val = abs(2.0 * N * np.max(conv) / (s * s))
    return val


def _construct_from_params(params):
    """Build the actual h array from parameters."""
    p = params[0]
    a = params[1]
    b = params[2]
    
    n_pos = int(round(p * N))
    n_pos = max(1, min(N - 1, n_pos))
    
    h = np.zeros(N)
    h[:n_pos] = a
    h[n_pos:] = b
    return h


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    # Start with the seed
    best_h = np.ones(N)
    best_val = 2.0
    
    # Try several initial parameter guesses
    initial_guesses = [
        [0.5, 1.0, -0.5],
        [0.5, 1.0, -1.0],
        [0.3, 1.0, -0.5],
        [0.7, 1.0, -0.5],
        [0.5, 2.0, -1.0],
        [0.4, 1.5, -0.5],
        [0.6, 1.5, -0.5],
        [0.5, 1.0, -0.3],
    ]
    
    bounds = [(0.05, 0.95), (0.1, 5.0), (-5.0, -0.01)]
    
    for guess in initial_guesses:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                _objective,
                guess,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            if result.success or result.fun < best_val:
                val = result.fun
                if val < best_val:
                    best_val = val
                    best_h = _construct_from_params(result.x)
        except Exception:
            continue
        
        if time.time() > deadline:
            break
    
    # Verify the result
    if not np.all(np.isfinite(best_h)):
        return np.ones(N)
    
    s = np.sum(best_h)
    if abs(s) < 1e-12:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
