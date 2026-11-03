"""New approach: Randomized Local Search for signed step functions."""
import numpy as np
import time

N = 400

def _compute_score(h):
    """
    Computes the objective function: max |autoconv(h)| / (sum h)^2.
    If sum h is close to 0, return a large penalty.
    """
    s = np.sum(h)
    if abs(s) < 1e-6:
        return 1e10
    # Compute autoconvolution using FFT
    # Zero-pad to avoid circular convolution
    size = 2 * N
    fft_h = np.fft.fft(h, n=size)
    autoconv = np.fft.ifft(fft_h * np.conj(fft_h)).real
    max_abs = np.max(np.abs(autoconv))
    return max_abs / (s * s)

def _local_search(h, max_iters=2000):
    """
    Performs local search by flipping individual elements.
    """
    current_h = h.copy()
    current_score = _compute_score(current_h)
    
    for _ in range(max_iters):
        # Pick a random index to flip
        idx = np.random.randint(N)
        old_val = current_h[idx]
        # Try changing to a few other values
        # Since we are dealing with "signed step heights", let's try flipping sign or changing to 0/1/-1
        # To keep it simple and fast, we'll try a small set of candidates
        candidates = [1, -1, 0, 2, -2]
        if old_val in candidates:
            candidates.remove(old_val)
        
        best_candidate = None
        best_candidate_score = current_score
        
        for c in candidates:
            current_h[idx] = c
            score = _compute_score(current_h)
            if score < best_candidate_score:
                best_candidate_score = score
                best_candidate = c
        
        # Revert if no improvement or keep best
        if best_candidate is not None:
            current_h[idx] = best_candidate
            current_score = best_candidate_score
        else:
            current_h[idx] = old_val
            
    return current_h, current_score

def construct():
    """
    Uses randomized local search to find a sequence minimizing the objective.
    Starts with a random initialization biased towards positive values to keep the sum high.
    """
    best_h = None
    best_score = float('inf')
    
    # Run a few random restarts
    start_time = time.time()
    max_time = 100.0  # Leave buffer for 120s limit
    
    while time.time() - start_time < max_time:
        # Initialize with mostly 1s, some -1s or 0s
        # To avoid sum=0, start with all 1s and perturb
        h = np.ones(N, dtype=int)
        # Randomly flip some to -1 or 0
        num_flips = np.random.randint(0, N // 2)
        if num_flips > 0:
            flip_indices = np.random.choice(N, num_flips, replace=False)
            h[flip_indices] = np.random.choice([-1, 0, 1], size=num_flips)
            
        h, score = _local_search(h, max_iters=1000)
        
        if score < best_score:
            best_score = score
            best_h = h.copy()
            
        # If we hit the theoretical bound, stop
        if best_score <= 1.0 / N * 1.001:
            break

    # Ensure we return a valid sequence
    if best_h is None:
        return np.ones(N)
    return best_h
