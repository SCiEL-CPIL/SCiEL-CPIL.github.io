"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Minimize the maximum overlap of h with shifted 1-h.
    Overlap metric: max_i h_i * (1 - h_{i+1}).
    Constraint: sum(h) = N/2, h_i in [0, 1].
    """
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initialize with constant 0.5 (analytic minimum for this metric)
    h = np.full(N, 0.5)
    best_h = h.copy()
    
    def calc_overlap(h_seq):
        shifted = np.roll(h_seq, -1)
        terms = h_seq * (1.0 - shifted)
        return np.max(terms)
    
    best_score = calc_overlap(best_h)
    
    # Local Search to ensure we aren't missing a better solution due to metric nuance
    rng = np.random.default_rng(42)
    
    # Number of iterations is large, but we break on time
    # We use a "step size" that decays or is random
    for it in range(200000):
        if time.time() > deadline:
            break
            
        # Select two indices to swap mass
        i = rng.integers(0, N)
        j = rng.integers(0, N)
        if i == j:
            continue
            
        # Max delta to keep in [0, 1]
        max_delta_up = 1.0 - h[i]
        max_delta_down = h[j]
        
        if max_delta_up <= 0 or max_delta_down <= 0:
            continue
            
        # Random delta
        # Use a scale that allows exploration
        # We try both positive and negative directions
        delta = rng.uniform(-max_delta_down, max_delta_up)
        
        # Apply
        h[i] += delta
        h[j] -= delta
        
        # Safety check
        if h[i] < 0 or h[i] > 1 or h[j] < 0 or h[j] > 1:
            h[i] -= delta
            h[j] += delta
            continue
            
        # Evaluate
        current_score = calc_overlap(h)
        
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
        else:
            # Revert
            h[i] -= delta
            h[j] += delta
            
    # Ensure exact sum and bounds for the returned result
    # The sum of 0.5s is exact. Perturbations might introduce FP error.
    # Adjust last element to fix sum.
    s = np.sum(best_h)
    target = N / 2.0
    diff = target - s
    best_h[-1] += diff
    
    # Clip to [0, 1] to be safe, though search should maintain it
    np.clip(best_h, 0.0, 1.0, out=best_h)
    
    # Re-adjust sum after clip (if clip changed anything)
    s = np.sum(best_h)
    diff = target - s
    best_h[-1] += diff
    np.clip(best_h, 0.0, 1.0, out=best_h)
    
    return list(best_h)
# EVOLVE-BLOCK-END
