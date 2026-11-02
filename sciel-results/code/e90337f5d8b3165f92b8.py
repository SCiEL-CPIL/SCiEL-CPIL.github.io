"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft

_start_time = time.time()
TIME_LIMIT = 110.0  # safety margin before 120s limit

def _autocorr_max_abs(x):
    """Compute maximum absolute value of autoconvolution of x."""
    n = len(x)
    # Use FFT for efficient autoconvolution
    size = 2 * n - 1
    # Pad to next power of 2 for efficiency
    fsize = 1
    while fsize < size:
        fsize *= 2
    X = fft(x, fsize)
    ac = np.real(ifft(X * np.conj(X)))[:size]
    return np.max(np.abs(ac))

def _score(x):
    """Score: maximum absolute autoconvolution."""
    return _autocorr_max_abs(x)

def _try_flip(x, i):
    """Try flipping element i, return new score if improved."""
    x[i] = -x[i]
    new_score = _autocorr_max_abs(x)
    x[i] = -x[i]  # revert
    return new_score

def _local_search(x, time_budget):
    """Local search: try single flips to minimize max abs autoconvolution."""
    current = _autocorr_max_abs(x)
    best = current
    deadline = time.time() + time_budget
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            break
        # Try a random position
        i = np.random.randint(0, N)
        x[i] = -x[i]
        new_score = _autocorr_max_abs(x)
        if new_score < best:
            best = new_score
            no_improve_count = 0
        else:
            x[i] = -x[i]  # revert
            no_improve_count += 1
    return x, best

def construct():
    global _start_time
    # Initialize with random ±1
    np.random.seed(42)
    
    best_x = None
    best_score = np.inf
    
    # Try a few different initializations
    time_per_restart = 8.0  # seconds per restart
    
    while time.time() - _start_time < TIME_LIMIT:
        # Random initialization
        x = np.random.choice([-1.0, 1.0], size=N)
        
        # Also try structured initialization
        if np.random.random() < 0.3:
            # Alternating pattern
            x = np.where(np.arange(N) % 2 == 0, 1.0, -1.0)
        elif np.random.random() < 0.15:
            # Block pattern
            block_size = np.random.randint(2, 20)
            for i in range(N):
                x[i] = 1.0 if (i // block_size) % 2 == 0 else -1.0
        
        # Local search
        remaining = TIME_LIMIT - (time.time() - _start_time)
        if remaining < 2.0:
            break
        budget = min(time_per_restart, remaining - 1.0)
        x, score = _local_search(x, budget)
        
        if score < best_score:
            best_score = score
            best_x = x.copy()
        
        # Check if we hit a good score
        if best_score <= 1.0:
            break
    
    if best_x is None:
        return np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END
