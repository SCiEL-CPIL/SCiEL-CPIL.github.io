"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _autocorr_peak(h):
    """
    Compute the maximum absolute autocorrelation over non-zero lags.
    This is the quantity we want to minimize for the autoconvolution upper bound.
    h is a sequence of signed step heights (typically +/-1).
    """
    # Use FFT for efficient computation of autocorrelation
    n_fft = 1 << (2 * N).bit_length()
    
    # Compute autocorrelation via FFT
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    r = np.fft.irfft(power, n=n_fft)
    
    # r[0] is lag 0 (sum of squares)
    # r[1..N-1] are lags 1..N-1
    # r[n_fft - (N-1) .. n_fft - 1] are lags -(N-1)..-1
    # We want max abs over lags 1..N-1 and -(N-1)..-1
    
    # Extract non-zero lags
    # lags 1 to N-1: r[1:N]
    # lags -(N-1) to -1: r[n_fft-N+1:n_fft]
    
    if N > 1:
        pos_lags = r[1:N]
        neg_lags = r[n_fft - N + 1 : n_fft]
        max_abs = max(np.max(np.abs(pos_lags)), np.max(np.abs(neg_lags)))
    else:
        max_abs = 0.0
    
    return max_abs

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    # Start with a random +/-1 sequence
    best_h = np.random.choice([-1, 1], size=N)
    best_score = _autocorr_peak(best_h)
    
    # Local search: flip random bits
    # We want to minimize the autocorrelation peak
    
    # Try a few different initializations and search strategies
    # Strategy 1: Simple random flips
    # Strategy 2: Greedy flips (flip bit if it reduces peak)
    
    current_h = best_h.copy()
    current_score = best_score
    
    # Greedy local search
    # For each position, try flipping and see if it reduces the peak
    # This is O(N^2 log N) per step if we recompute, but we can be smarter
    
    # Actually, let's use a simpler approach:
    # Random restarts with greedy refinement
    
    rng = np.random.RandomState(42)
    
    for restart in range(100):
        if time.time() > deadline:
            break
        
        # Random initialization
        h = rng.choice([-1, 1], size=N).astype(np.float64)
        score = _autocorr_peak(h)
        
        # Greedy refinement
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try flipping each position
            for i in range(N):
                if time.time() > deadline:
                    break
                h[i] = -h[i]
                new_score = _autocorr_peak(h)
                if new_score < score:
                    score = new_score
                    improved = True
                    if score < best_score:
                        best_score = score
                        best_h = h.copy()
                else:
                    h[i] = -h[i]
        
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    return best_h
# EVOLVE-BLOCK-END
