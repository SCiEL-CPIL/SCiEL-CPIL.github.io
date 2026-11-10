"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 28.0  # Leave 2s margin for safety
    
    best_set = None
    best_size = N + 1
    
    # We will run randomized greedy saturation.
    # Each run is O(N^2) which is very fast for N=256.
    
    while time.time() - start_time < time_limit:
        selected = []
        # Use a boolean array for O(1) sum lookups
        # Max sum is 256 + 255 = 511. Index 0 unused.
        sums = [False] * (2 * N + 1)
        
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            # Check if x can be added
            # 1. Check 2*x
            if sums[2 * x]:
                continue
            
            # 2. Check x + a for all a in selected
            valid = True
            for a in selected:
                if sums[x + a]:
                    valid = False
                    break
            
            if valid:
                selected.append(x)
                sums[2 * x] = True
                for a in selected[:-1]:  # Mark sums with previous elements
                    sums[x + a] = True
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected[:]
            
            # Optional: If we find a very small set (e.g., <= 16), we might stop
            # but to be safe and find potentially smaller, we continue.
            # However, if best_size is 15, it's likely optimal or near-optimal.
            # We'll just let the time limit handle it.

    return best_set if best_set else [1]
# EVOLVE-BLOCK-END
