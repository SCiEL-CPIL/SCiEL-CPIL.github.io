import random
import time
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

def max_sidon_size(arr):
    n = len(arr)
    # Check subsets in decreasing order of size
    # Return the size of the largest Sidon subset
    for k in range(n, 0, -1):
        # Generate combinations of indices of size k
        for combo in combinations(range(n), k):
            subset = [arr[i] for i in combo]
            if is_sidon(subset):
                return k
    return 0

def construct():
    start_time = time.time()
    # Safety margin: aim to finish by 55s to be safe against 60s/90s limits
    deadline = start_time + 55.0
    
    # Initial best
    best_set = list(range(12))
    best_val = max_sidon_size(best_set)
    
    random.seed(42)
    
    # Search in small ranges to force collisions
    # Knowledge says [0, 25] or [0, 30] is good.
    search_ranges = [20, 25, 30, 35, 40]
    
    for rng_size in search_ranges:
        if time.time() > deadline:
            break
            
        # Do some random restarts for this range
        max_restarts = 5000
        for _ in range(max_restarts):
            if time.time() > deadline:
                break
            
            current_set = random.sample(range(rng_size), 12)
            current_val = max_sidon_size(current_set)
            
            if current_val < best_val:
                best_val = current_val
                best_set = current_set[:]
                if best_val <= 4:
                    return best_set
                
            # Simple local search: try to improve current_set
            for step in range(20):
                if time.time() > deadline:
                    break
                i = random.randint(0, 11)
                old_val = current_set[i]
                new_val = random.randint(0, rng_size - 1)
                if new_val in current_set:
                    continue
                
                current_set[i] = new_val
                new_val_result = max_sidon_size(current_set)
                
                if new_val_result < current_val:
                    current_val = new_val_result
                    if current_val < best_val:
                        best_val = current_val
                        best_set = current_set[:]
                        if best_val <= 4:
                            return best_set
                else:
                    # Revert if not better
                    current_set[i] = old_val
                    
    return best_set
