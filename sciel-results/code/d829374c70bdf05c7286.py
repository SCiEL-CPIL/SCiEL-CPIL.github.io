"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.fft import fft, ifft
import time

def _autoconv_peak(x):
    """Compute max absolute value of linear autoconvolution."""
    # Use FFT for efficient autoconvolution
    n = len(x)
    size = 2 * n
    X = fft(x, n=size)
    ac = np.real(ifft(X * X, n=size))[:n]
    return np.max(np.abs(ac))

def _score(x):
    """Compute the C3' value: max|autoconv| / (sum(x))^2"""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    peak = _autoconv_peak(x)
    return peak / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Start with random signs
    rng = np.random.default_rng(42)
    
    best_x = None
    best_score = np.inf
    
    # Try several random initializations
    for trial in range(5):
        if time.time() > deadline - 20.0:
            break
            
        x = rng.choice([-1.0, 1.0], size=N)
        
        # Local search: flip individual elements
        current_score = _score(x)
        if current_score < best_score:
            best_score = current_score
            best_x = x.copy()
        
        # Iterative improvement
        improved = True
        while improved and time.time() < deadline - 15.0:
            improved = False
            # Randomly sample positions to try flipping
            idxs = rng.permutation(N)
            for i in idxs:
                if time.time() > deadline - 15.0:
                    break
                x[i] *= -1
                new_score = _score(x)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    x[i] *= -1  # revert
            
            if current_score < best_score:
                best_score = current_score
                best_x = x.copy()
    
    # Final polish: more thorough local search on best solution
    if best_x is not None and time.time() < deadline - 10.0:
        x = best_x.copy()
        current_score = best_score
        for _ in range(50):
            if time.time() > deadline - 10.0:
                break
            improved = False
            idxs = rng.permutation(N)
            for i in idxs:
                if time.time() > deadline - 10.0:
                    break
                x[i] *= -1
                new_score = _score(x)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    x[i] *= -1
            if not improved:
                break
            if current_score < best_score:
                best_score = current_score
                best_x = x.copy()
    
    if best_x is None:
        return np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END
