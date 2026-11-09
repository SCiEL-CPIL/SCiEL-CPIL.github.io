"""Optimized solver for low-discrepancy on all arithmetic progressions."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _max_ap_abs_sum(signs, n=256):
    """
    Compute the maximum absolute signed sum over all arithmetic progressions
    contained in [1, n].
    
    An arithmetic progression is defined by start s (1-indexed), step d (positive integer),
    and length L >= 1, such that s, s+d, ..., s+(L-1)d <= n.
    
    We iterate over all steps d and starts s, and for each, compute the maximum
    absolute partial sum.
    """
    max_abs = 0
    # Precompute 0-indexed array
    a = signs  # list of -1 or 1, length n
    
    # For each step d
    for d in range(1, n + 1):
        # For each start s (0-indexed: 0 to n-1), but we only need starts where s+d*(L-1) < n
        # Actually, for a fixed d, the progression starts at s and goes s, s+d, s+2d, ...
        # The number of elements is floor((n-1-s)/d) + 1 if s < n
        # We need to compute max absolute partial sum for each such progression
        
        # To be efficient, for each residue class mod d, we can process the sequence
        # But since d can be up to n, and n=256, total work is sum_{d=1}^{n} n = O(n^2)
        # which is 256^2 = 65536, very fast.
        
        for s in range(d):  # start positions 0, 1, ..., d-1 (0-indexed)
            # The progression is s, s+d, s+2d, ... < n
            # Compute partial sums
            curr_sum = 0
            for idx in range(s, n, d):
                curr_sum += a[idx]
                abs_val = abs(curr_sum)
                if abs_val > max_abs:
                    max_abs = abs_val
    return max_abs

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = 256
    rng = random.Random(42)
    
    # Initial random solution
    best_signs = [rng.choice((-1, 1)) for _ in range(n)]
    best_score = _max_ap_abs_sum(best_signs, n)
    
    # Local search with simulated annealing or simple hill climbing
    # Try to improve by flipping single signs
    
    current_signs = best_signs[:]
    current_score = best_score
    
    # Hill climbing with restarts
    max_iterations = 500000  # We'll check time each iteration
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Pick a random position to flip
        pos = rng.randint(0, n - 1)
        
        # Flip it
        old_val = current_signs[pos]
        current_signs[pos] = -old_val
        
        # Evaluate new score
        new_score = _max_ap_abs_sum(current_signs, n)
        
        if new_score < current_score:
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_signs = current_signs[:]
        else:
            # Revert
            current_signs[pos] = old_val
    
    return best_signs
# EVOLVE-BLOCK-END
