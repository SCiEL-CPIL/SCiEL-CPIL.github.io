"""
Optimization of signed step function to minimize autoconvolution upper bound C3.
"""

import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.linalg import eigh

N = 400

def _compute_c3(h):
    """Compute the upper bound on C3 given signed step heights h.
    
    The autoconvolution of a step function is related to the sum of products
    of adjacent values. We minimize the maximum absolute value of the 
    autoconvolution, which serves as an upper bound on C3.
    
    For a step function with heights h_i on intervals, the autoconvolution
    at lag k is sum_i h_i * h_{i+k}. The maximum of these values (in absolute
    sense) gives the upper bound.
    """
    # Compute all lags of autoconvolution
    # h has length N
    # autoconv[k] = sum_{i=0}^{N-1-k} h[i] * h[i+k]
    # We want to minimize max_k |autoconv[k]|
    
    # Vectorized computation of autoconvolution
    # Using FFT for efficiency
    h_padded = np.zeros(2 * N)
    h_padded[:N] = h
    fft_h = np.fft.fft(h_padded, n=2*N)
    autoconv = np.fft.ifft(fft_h * np.conj(fft_h)).real[:N]
    
    # The upper bound is the maximum absolute value of the autoconvolution
    # Note: autoconv[0] = sum h_i^2, which is fixed if h_i in {-1, 1}
    # For signed steps, we typically consider h_i in {-1, 1}
    # The bound C3 is related to max |autoconv[k]| for k >= 1
    
    # Return the maximum absolute value of non-zero lag autoconvolutions
    if len(autoconv) > 1:
        return np.max(np.abs(autoconv[1:]))
    return np.max(np.abs(autoconv))

def construct():
    """Construct a signed step function with N intervals to minimize C3."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initialize with random signed step function
    best_h = np.random.choice([-1.0, 1.0], size=N)
    best_score = _compute_c3(best_h)
    
    # Local search with multi-flip moves
    # Use a simple iterative improvement approach
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            # Try flipping h[i]
            h_trial = best_h.copy()
            h_trial[i] = -h_trial[i]
            score = _compute_c3(h_trial)
            if score < best_score - 1e-12:
                best_h = h_trial
                best_score = score
                improved = True
    
    # Try 2-flip moves
    while time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            for j in range(i+1, min(i+50, N)):  # Limit to nearby pairs for efficiency
                h_trial = best_h.copy()
                h_trial[i] = -h_trial[i]
                h_trial[j] = -h_trial[j]
                score = _compute_c3(h_trial)
                if score < best_score - 1e-12:
                    best_h = h_trial
                    best_score = score
                    improved = True
                    break
            if improved:
                break
    
    return best_h
