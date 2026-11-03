import numpy as np
import time

N = 100

def calculate_score(h):
    # h is an array of length N
    # Overlap at shift k: sum(h[i] * (1 - h[(i+k)%N]))
    # We want to minimize the max overlap
    max_overlap = 0
    for k in range(1, N):
        # Vectorized calculation
        # h_shifted = np.roll(h, k)
        # overlap = np.sum(h * (1 - h_shifted))
        # Using dot product for speed
        # sum(h_i * (1 - h_{i+k})) = sum(h) - sum(h_i h_{i+k})
        # sum(h) is N/2
        # autocorr = np.sum(h * np.roll(h, k))
        # overlap = N/2 - autocorr
        pass
    # This loop is slow in Python. Use FFT?
    # Autocorrelation can be computed via FFT.
    # But we need max over k.
    # Let's just use the vectorized numpy roll in a loop for N=100, it's fast enough.
    max_ov = 0.0
    for k in range(1, N):
        h_shifted = np.roll(h, k)
        ov = np.sum(h * (1.0 - h_shifted))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def construct():
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    # Initial guess: uniform
    h_best = np.full(N, 0.5)
    score_best = calculate_score(h_best)
    
    # Try a few random perturbations or local search
    # Since 0.5 is likely optimal, we might not find anything better.
    # But let's try a binary pattern just in case the metric favors it.
    # Or a sinusoidal pattern.
    
    # Try sinusoidal: h_i = 0.5 + A * sin(2*pi*f*i/N)
    # A must be such that h_i in [0,1]. Max A = 0.5.
    # Sum must be N/2. Sine has mean 0, so sum is N/2.
    
    best_h = h_best
    best_score = score_best
    
    # Check deadline
    if time.time() > deadline:
        return best_h
    
    # Try some frequencies
    for f in range(1, N//2):
        if time.time() > deadline:
            break
        for A in [0.1, 0.3, 0.5]:
            h = 0.5 + A * np.sin(2 * np.pi * f * np.arange(N) / N)
            # Clip to [0,1]
            h = np.clip(h, 0, 1)
            # Rescale to ensure sum is N/2?
            # Clipping changes the sum.
            # If we clip, sum might not be N/2.
            # We need to adjust.
            # Simple adjustment: shift by constant?
            # Or just accept the score if it's valid?
            # The task requires sum exactly len/2.
            # So we must normalize.
            # But normalization might break the [0,1] bound if we just scale.
            # Let's just use the unclipped version if A is small enough.
            # For A=0.5, min is 0, max is 1. Sum is N/2.
            # So for A=0.5, it is valid.
            if A <= 0.5:
                s = np.sum(h)
                if abs(s - N/2) > 1e-9:
                    # Adjust by adding a constant?
                    # h + c. Sum becomes N/2.
                    # h_i + c in [0,1]?
                    # If h is 0.5 +/- 0.5 sin, min 0, max 1.
                    # Adding c shifts it.
                    # If sum is not N/2, it's due to numerical error or clipping.
                    # For sine, sum is exactly 0 (mean).
                    # So sum is N/2.
                    pass
                
                score = calculate_score(h)
                if score < best_score:
                    best_score = score
                    best_h = h
    
    return best_h
