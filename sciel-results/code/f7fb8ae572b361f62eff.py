"""Optimized subset-sum construction via simulated annealing / local search."""
# EVOLVE-BLOCK-START
import random
import time
import sys
import math

N = 12
MAX_VAL = 1048576

def _all_subset_sums(arr):
    """Compute all subset sums."""
    sums = [0] * (1 << N)
    for i in range(N):
        bit = 1 << i
        for mask in range(bit):
            sums[bit | mask] = sums[mask] + arr[i]
    return sums

def _count_collisions(arr):
    """Count number of collisions (pairs with same sum)."""
    sums = _all_subset_sums(arr)
    unique = set(sums)
    return len(sums) - len(unique)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    # Start with a known-good superincreasing sequence (powers of 2)
    arr = [1 << i for i in range(N)]
    best_arr = arr[:]
    best_max = max(arr)
    
    random.seed(42)
    
    # Simulated Annealing
    T = 50.0
    cooling = 0.999
    
    # Objective: max(arr) + 1000 * collisions
    
    current_max = max(arr)
    current_coll = 0
    current_cost = current_max + 1000 * current_coll
    
    while time.time() < deadline and T > 0.1:
        # Choose index to modify
        # Bias towards the largest element to reduce it
        if random.random() < 0.7:
            idx = max(range(N), key=lambda i: arr[i])
        else:
            idx = random.randint(0, N - 1)
            
        old_val = arr[idx]
        
        # Generate new value
        # Try to decrease, but allow some increase to escape local minima
        # Range: [1, best_max]
        step = random.randint(1, max(1, old_val // 10))
        if random.random() < 0.8:
            # Try to decrease
            new_val = max(1, old_val - step)
        else:
            # Try to increase or random
            new_val = min(best_max, old_val + step)
            
        if new_val == old_val:
            continue
            
        # Check if new_val duplicates another element
        # If arr has duplicates, it's invalid.
        # We must ensure distinctness.
        if new_val in arr and arr[idx] != new_val:
             continue
             
        arr[idx] = new_val
        
        new_max = max(arr)
        new_coll = _count_collisions(arr)
        new_cost = new_max + 1000 * new_coll
        
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            current_max = new_max
            current_coll = new_coll
            current_cost = new_cost
            
            if new_coll == 0 and new_max < best_max:
                best_max = new_max
                best_arr = arr[:]
        else:
            # Reject
            arr[idx] = old_val
            
        T *= cooling
        
    # Final verification and cleanup
    # Ensure best_arr is valid
    if _count_collisions(best_arr) != 0 or len(set(best_arr)) != N:
        # Fallback to powers of 2
        best_arr = [1 << i for i in range(N)]
        
    return best_arr
# EVOLVE-BLOCK-END
