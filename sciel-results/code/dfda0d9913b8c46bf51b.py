"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import fftconvolve

def _autoconv_max_abs(x):
    """Compute maximum absolute value of autoconvolution of x."""
    # Autoconvolution is convolution of x with reversed x
    rev = x[::-1]
    conv = fftconvolve(x, rev, mode='full')
    return np.max(np.abs(conv))

def _squared_integral(x):
    """Compute squared integral (sum of x) squared."""
    s = np.sum(x)
    return s * s

def _score(x):
    """Compute the C3' score: max_abs_autoconv / squared_integral."""
    sq_int = _squared_integral(x)
    if sq_int == 0:
        return float('inf')
    return _autoconv_max_abs(x) / sq_int

def _random_sparse_signed(n, density, rng):
    """Generate a random sparse signed sequence."""
    x = np.zeros(n)
    mask = rng.random(n) < density
    signs = rng.choice([-1.0, 1.0], size=n)
    x[mask] = signs[mask]
    return x

def _local_sign_flip_optimize(x, max_iters=5000, time_limit=8.0):
    """
    Local optimization via sign flips on non-zero entries.
    Tries flipping signs to reduce the max absolute autoconvolution.
    """
    x = x.copy()
    nonzero_idx = np.nonzero(x)[0]
    if len(nonzero_idx) == 0:
        return x
    
    current_score = _score(x)
    best_x = x.copy()
    best_score = current_score
    
    start_time = time.time()
    no_improve = 0
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Randomly select a non-zero index to flip
        idx = nonzero_idx[rng.randint(len(nonzero_idx))]
        old_val = x[idx]
        x[idx] = -old_val
        
        new_score = _score(x)
        
        if new_score < current_score - 1e-12:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_x = x.copy()
            no_improve = 0
        else:
            x[idx] = old_val  # revert
            no_improve += 1
            if no_improve > 200:
                # Try a random restart of the local search state
                no_improve = 0
    
    return best_x

def construct():
    """
    Construct a signed step function sequence minimizing max_abs_autoconv / squared_integral.
    
    Strategy: Use sparse random signed sequences with local sign-flip optimization.
    The sparsity reduces the autoconvolution peak relative to the integral.
    """
    global rng
    rng = np.random.default_rng(42)
    
    deadline = 110.0  # Safety margin before 120s limit
    start_time = time.time()
    
    best_x = None
    best_score = float('inf')
    
    # Try multiple densities
    densities = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    
    for density in densities:
        if time.time() - start_time > deadline:
            break
        
        # Multiple trials per density
        n_trials = 3
        for trial in range(n_trials):
            if time.time() - start_time > deadline:
                break
            
            x = _random_sparse_signed(N, density, rng)
            
            # Quick local optimization
            x_opt = _local_sign_flip_optimize(x, max_iters=3000, time_limit=5.0)
            
            s = _score(x_opt)
            if s < best_score:
                best_score = s
                best_x = x_opt.copy()
    
    # If no valid result found, fall back to all-ones
    if best_x is None:
        best_x = np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END
