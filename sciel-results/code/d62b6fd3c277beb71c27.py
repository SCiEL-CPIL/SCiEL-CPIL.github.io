import time
import random
from itertools import combinations

def is_sidon(subset):
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def largest_sidon_subset_size(S):
    S = sorted(S)
    n = len(S)
    for k in range(n, 0, -1):
        for combo in combinations(S, k):
            if is_sidon(combo):
                return k
    return 1

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    # Start with range(12)
    current_set = list(range(12))
    current_val = largest_sidon_subset_size(current_set)
    best_set = current_set[:]
    best_val = current_val
    
    # Local search: try to replace one element
    # Focus on small numbers [0, 50]
    
    # To avoid getting stuck, use simulated annealing or random restarts
    # But given the small size, simple hill climbing with random restarts is fine.
    
    # We will try to modify the current best set.
    
    # Strategy: 
    # 1. Try all single-element replacements in [0, 128] for the current best.
    # 2. If no improvement, try random perturbations.
    
    # However, checking all 128 replacements for each of 12 positions is 1536 checks.
    # Each check involves computing largest Sidon subset.
    # This is fast.
    
    # Let's do a few iterations of this.
    
    for iteration in range(100):
        if time.time() > deadline:
            break
            
        improved = False
        
        # Try replacing each element with a random other in [0, 128]
        # Or better, try all small values [0, 30]
        
        for i in range(12):
            if time.time() > deadline:
                break
                
            old_val = best_set[i]
            
            # Try new values
            # We can sample or iterate
            # Iterating 0..128 is 129 * 12 = 1548 checks.
            # 1548 * (cost of largest_sidon)
            # Cost is ~1000 is_sidon checks.
            # 1.5M is_sidon checks.
            # is_sidon is ~15 ops.
            # 20M ops.
            # Python might take a few seconds.
            # So 100 iterations might be too slow.
            
            # Let's reduce the range of new values to [0, 20]
            # And only do 10 iterations.
            
            for new_val in range(0, 20):
                if new_val == old_val:
                    continue
                    
                # Check if new_val is already in the set
                if new_val in best_set:
                    continue
                    
                # Create new set
                new_set = best_set[:]
                new_set[i] = new_val
                
                # Check validity (distinct)
                # We ensured new_val not in best_set, and best_set has distinct.
                # So new_set has distinct.
                
                val = largest_sidon_subset_size(new_set)
                
                if val < best_val:
                    best_val = val
                    best_set = new_set
                    improved = True
                    # Restart the loop to update i? 
                    # Or just continue.
                    # If we improve, we should probably restart the search from this new best.
                    # But breaking out of inner loops is messy.
                    # Just update and continue.
                    
        if not improved:
            # Try a random restart or jump
            # Pick a random set from [0, 30]
            if random.random() < 0.1:
                cand = random.sample(range(0, 30), 12)
                val = largest_sidon_subset_size(cand)
                if val < best_val:
                    best_val = val
                    best_set = cand[:]

    return best_set
