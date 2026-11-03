"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _score_seq(x):
    """
    Compute the metric: max absolute autoconvolution (excluding zero lag) / (integral)^2.
    For a sequence of signed steps of width 1, the 'integral' is sum(x).
    Autoconvolution at lag k is sum_{i} x[i] * x[i+k].
    We want to minimize the max over k != 0 of |autoconv[k]| / (sum(x))^2.
    """
    # Use FFT for efficient autoconvolution
    n = len(x)
    # Zero-pad to next power of 2 for efficient FFT
    m = 1
    while m < 2 * n:
        m *= 2
    F = np.fft.rfft(x, m)
    G = F * np.conj(F)
    ac = np.fft.irfft(G, m)
    # ac[0] is the zero-lag (sum of squares)
    # ac[1:n] are lags 1 to n-1
    # ac[n:m] are negative lags (mirror of ac[1:n])
    # We care about lags 1 to n-1
    sidelobes = ac[1:n]
    max_abs = np.max(np.abs(sidelobes))
    integral_sq = np.sum(x) ** 2
    if integral_sq == 0:
        return 1e10  # Degenerate case
    return max_abs / integral_sq

def _local_search(x, max_iters=5000):
    """
    Local search: flip bits to reduce the score.
    """
    current = x.copy().astype(np.float64)
    current_score = _score_seq(current)
    
    for _ in range(max_iters):
        # Randomly pick a position to flip
        i = np.random.randint(N)
        old_val = current[i]
        new_val = -old_val
        
        # Quick check: flipping changes the integral.
        # If sum was S, new sum is S - 2*old_val.
        # If new sum is 0, the denominator becomes 0, which is bad (infinite score).
        # Actually, if sum is 0, the metric is undefined or infinite. We should avoid this.
        # Let's check the new sum.
        current_sum = np.sum(current)
        new_sum = current_sum - 2 * old_val
        
        # If new_sum is 0, skip this flip (or penalize heavily)
        if abs(new_sum) < 1e-6:
            continue
            
        current[i] = new_val
        new_score = _score_seq(current)
        
        if new_score < current_score:
            current_score = new_score
        else:
            # Revert
            current[i] = old_val
            
    return current, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    best_seq = None
    best_score = 1e10
    
    # Initialize with random sequence
    np.random.seed(42)
    
    # Try a few different random starts
    num_starts = 50
    
    for s in range(num_starts):
        if time.time() > deadline:
            break
            
        x = np.random.choice([-1.0, 1.0], size=N)
        
        # Ensure the sum is not zero. If it is, flip one bit.
        sum_x = np.sum(x)
        if abs(sum_x) < 1e-6:
            x[0] = -x[0]
            
        seq, score = _local_search(x, max_iters=1000)
        
        if score < best_score:
            best_score = score
            best_seq = seq.copy()
            
    # If we still have time, do more local search on the best sequence found
    if time.time() < deadline and best_seq is not None:
        # Try to improve further with more iterations
        seq, score = _local_search(best_seq, max_iters=5000)
        if score < best_score:
            best_score = score
            best_seq = seq.copy()

    # Final check: ensure the result is valid (non-zero sum)
    if best_seq is None or abs(np.sum(best_seq)) < 1e-6:
        # Fallback to a known good sequence (e.g., all ones)
        return np.ones(N)
        
    return best_seq

# EVOLVE-BLOCK-END
