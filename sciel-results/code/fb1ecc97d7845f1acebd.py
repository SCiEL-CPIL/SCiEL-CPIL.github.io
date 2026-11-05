"""Program for low discrepancy on all arithmetic progressions.
Uses a greedy/iterative improvement approach with NumPy for speed.
"""
import numpy as np

def construct():
    n = 256
    # Start with a deterministic pattern that tends to have low discrepancy:
    # e.g., Thue-Morse-like or simple alternating, then improve.
    # Let's start with alternating signs as a reasonable baseline.
    signs = np.array([1 if i % 2 == 0 else -1 for i in range(n)], dtype=np.int8)
    
    # Precompute all arithmetic progressions as index lists
    # For efficiency, we'll group progressions by (start, step) and compute sums efficiently.
    # But there are O(n^2) progressions, which is ~65536, manageable.
    
    # Actually, let's use a smarter approach: 
    # The maximum discrepancy over all APs can be estimated by checking a subset,
    # but to minimize it, we can do local search: flip a sign if it reduces the max discrepancy.
    
    # For speed, we'll maintain a data structure to quickly compute AP sums.
    # An AP is defined by (start, step, length). The sum is sum of signs[start-1 : start-1+length*step : step].
    
    # Let's precompute all APs as (start_idx, step, length) where start_idx is 0-based.
    # But storing all is memory-heavy. Instead, we can iterate over them on the fly.
    
    # Better: use NumPy to compute all AP sums at once? Not directly, since steps vary.
    
    # Alternative: iterative local search. At each step, try flipping each position and 
    # compute the new max discrepancy. Choose the flip that gives the best improvement.
    # This is O(n^2) per iteration if done naively, but we can be smarter.
    
    # Let's try a simpler heuristic: 
    # 1. Compute current max discrepancy.
    # 2. For each position i, compute the "contribution" of flipping it.
    # 3. Flip the position that most reduces the max discrepancy.
    
    # To compute max discrepancy efficiently, we need to check all APs.
    # Number of APs: for each start (1..n), each step (1..n-start+1), each length (1..(n-start+1)//step + 1).
    # Total ~ sum_{s=1}^{n} sum_{d=1}^{n-s+1} (n-s+1)//d ~ O(n^2 log n) ≈ 256^2 * 6 ≈ 400k. Manageable.
    
    def max_discrepancy(signs):
        n = len(signs)
        max_abs = 0
        for start in range(n):
            for step in range(1, n - start):
                # Compute sum for all lengths of this AP
                s = 0
                for length in range(1, (n - start) // step + 1):
                    s += signs[start + (length - 1) * step]
                    abs_s = abs(s)
                    if abs_s > max_abs:
                        max_abs = abs_s
        return max_abs
    
    # Initial max discrepancy
    current_max = max_discrepancy(signs)
    
    # Local search: try flipping each sign, see if it improves
    improved = True
    while improved:
        improved = False
        best_flip = -1
        best_new_max = current_max
        
        for i in range(n):
            # Flip sign at position i
            signs[i] = -signs[i]
            new_max = max_discrepancy(signs)
            # Unflip
            signs[i] = -signs[i]
            
            if new_max < best_new_max:
                best_new_max = new_max
                best_flip = i
        
        if best_flip >= 0:
            signs[best_flip] = -signs[best_flip]
            current_max = best_new_max
            improved = True
    
    return [int(s) for s in signs]
