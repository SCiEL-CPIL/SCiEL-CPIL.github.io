import numpy as np
import time

N = 100
DELTA = 2.0 / N

def compute_max_overlap_circular(h):
    # h is an array of length N
    # Compute overlap for all shifts k
    # O_k = sum_i h_i * (1 - h_{i+k}) * DELTA
    # This is equivalent to DELTA * (sum(h) - sum(h * roll(h, k)))
    # sum(h) is N/2
    # We want to minimize max O_k
    # O_k = N/2 * DELTA - DELTA * sum(h * roll(h, k))
    # = 1 - DELTA * autocorr_k
    # Minimizing max O_k is maximizing min autocorr_k
    
    # Vectorized autocorrelation
    # Use FFT for speed? N=100 is small, direct sum is fine.
    # But we need max over k.
    
    # Let's just compute the matrix of overlaps or loop.
    # Loop is O(N^2) = 10,000 ops. Very fast.
    
    max_overlap = 0.0
    for k in range(N):
        shifted_h = np.roll(h, k)
        overlap = np.sum(h * (1.0 - shifted_h)) * DELTA
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial guess: constant 0.5
    best_h = np.full(N, 0.5)
    best_score = compute_max_overlap_circular(best_h)
    
    # Try a few random restarts and simple hill climbing
    # Since N=100, we can afford some iterations.
    
    rng = np.random.default_rng(42)
    
    for i in range(1000): # Max iterations
        if time.time() > deadline:
            break
        
        # Random perturbation
        h = best_h.copy()
        # Add noise
        noise = rng.normal(0, 0.01, N)
        h += noise
        h = np.clip(h, 0, 1)
        
        # Re-normalize to sum N/2
        # This is tricky. Sum must be exactly N/2.
        # Simple scaling might violate bounds.
        # We can use a projection method or just accept if close?
        # The task says "sum is exactly len/2".
        # So we must enforce this.
        
        # Method: Add noise, then adjust to fix sum while respecting bounds?
        # Or just use a valid move: swap mass between two bins.
        
        # Let's try a coordinate descent style step.
        # Pick two indices i, j. Move mass from i to j.
        # h[i] - eps, h[j] + eps.
        # Ensure bounds.
        
        idx1 = rng.integers(N)
        idx2 = rng.integers(N)
        if idx1 == idx2:
            continue
            
        # Determine max possible move
        # h[idx1] can decrease by h[idx1]
        # h[idx2] can increase by 1 - h[idx2]
        max_move = min(h[idx1], 1.0 - h[idx2])
        
        if max_move <= 0:
            continue
            
        # Try a few step sizes
        for factor in [0.1, 0.5, 1.0]:
            move = max_move * factor
            h_new = h.copy()
            h_new[idx1] -= move
            h_new[idx2] += move
            
            score = compute_max_overlap_circular(h_new)
            if score < best_score:
                best_score = score
                best_h = h_new
                break # Accept best improvement for this pair? Or continue?
                # Let's accept the first improvement to keep it simple
        
    return best_h
