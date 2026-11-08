"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _overlaps(h):
    H = np.fft.rfft(h, n=2*N)
    autocorr = np.fft.irfft(H * np.conj(H), n=2*N)[:N]
    sum_h = np.sum(h)
    overlaps = sum_h - autocorr
    return overlaps

def _max_overlap(h):
    return np.max(_overlaps(h))

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)  # 25.0
    
    # Gradient-based local search
    h = best_h.copy()
    lr = 0.1
    
    for step in range(50000):
        if time.time() > deadline:
            break
        
        overlaps = _overlaps(h)
        max_k = np.argmax(overlaps)
        
        # Gradient of overlap_k w.r.t. h[j]:
        # overlap_k = sum_i h[i] - sum_i h[i]*h[i+k]
        # d(overlap_k)/d(h[j]) = 1 - (h[j-k] + h[j+k]) ... wait
        # d/d(h[j]) [sum_i h[i]*h[i+k]] = h[j-k] + h[j+k] (if j-k and j+k are in range, circularly)
        # So gradient of overlap_k w.r.t. h[j] = 1 - h[(j-k) mod N] - h[(j+k) mod N]
        
        k = max_k
        grad = np.ones(N) - np.roll(h, -k) - np.roll(h, k)
        
        # Update: decrease h where gradient is positive (overlap increases with h)
        # We want to decrease overlap_k, so move h in direction -grad
        h_new = h - lr * grad
        h_new = np.clip(h_new, 0, 1)
        
        # Re-normalize to sum = N/2
        current_sum = np.sum(h_new)
        if current_sum > 0:
            h_new = h_new * (N/2) / current_sum
            h_new = np.clip(h_new, 0, 1)
            # After clipping, re-normalize
            current_sum = np.sum(h_new)
            if current_sum > 0:
                h_new = h_new * (N/2) / current_sum
                h_new = np.clip(h_new, 0, 1)
        
        score = _max_overlap(h_new)
        if score < best_score:
            best_h = h_new.copy()
            best_score = score
            lr = min(lr * 1.05, 0.5)
        else:
            lr = max(lr * 0.9, 0.01)
            h = h_new  # Still update to explore
    
    return best_h
# EVOLVE-BLOCK-END
