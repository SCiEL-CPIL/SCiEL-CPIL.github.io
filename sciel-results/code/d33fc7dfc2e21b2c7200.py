"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal import fftconvolve, remez
import time

def _autoconv(x):
    """Full autoconvolution of x."""
    return fftconvolve(x, x[::-1], mode='full')

def _score(x):
    """Compute max|autoconv| / (sum)^2."""
    s = x.sum()
    if abs(s) < 1e-12:
        return float('inf')
    ac = _autoconv(x)
    return float(np.max(np.abs(ac)) / (s * s))

def _remez_init(n, cutoff_frac, num_riple):
    """Design an equiripple filter using Parks-McClellan."""
    try:
        # Design a lowpass filter with equiripple magnitude
        # numtaps = n, bands = [0, cutoff, 1], desired = [1, 0]
        b = remez(n, [0, cutoff_frac, 1.0], [1, 0], num_riple)
        return b
    except Exception:
        return np.ones(n) / n

def construct():
    start = time.time()
    deadline = start + 135.0  # safety margin
    
    best_x = np.ones(N, dtype=np.int64)
    best_score = _score(best_x)
    
    rng = np.random.default_rng(123)
    
    # Try multiple Remez designs with different parameters
    cutoffs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ripple_counts = [5, 10, 15, 20]
    
    for cutoff in cutoffs:
        for num_riple in ripple_counts:
            if time.time() > deadline - 30.0:
                break
            
            x = _remez_init(N, cutoff, num_riple)
            
            # Scale to have positive sum
            if x.sum() < 0:
                x = -x
            
            # Round to integers
            x_int = np.round(x).astype(np.int64)
            
            # Ensure non-zero sum
            if abs(x_int.sum()) < 10:
                x_int = np.ones(N, dtype=np.int64)
            
            cur_score = _score(x_int)
            
            if cur_score < best_score:
                best_score = cur_score
                best_x = x_int.copy()
    
    # Local search refinement from best Remez solution
    if time.time() < deadline - 20.0:
        x = best_x.astype(np.float64)
        cur_score = best_score
        
        max_iters = 3000
        iters = 0
        while iters < max_iters:
            iters += 1
            if time.time() > deadline - 10.0:
                break
            
            # Try flipping a random element
            idx = rng.integers(0, N)
            x[idx] *= -1
            new_score = _score(x)
            if new_score < cur_score:
                cur_score = new_score
            else:
                x[idx] *= -1  # revert
            
            # Periodically try larger moves
            if iters % 50 == 0:
                idx = rng.integers(0, N)
                x[idx] += 1
                new_score = _score(x)
                if new_score < cur_score:
                    cur_score = new_score
                else:
                    x[idx] -= 1  # revert
        
        if cur_score < best_score:
            best_score = cur_score
            best_x = np.round(x).astype(np.int64)
    
    return best_x
# EVOLVE-BLOCK-END
