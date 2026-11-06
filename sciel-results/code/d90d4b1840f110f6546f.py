"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 85  # Safety margin
    
    def subset_sums_set(arr):
        """Compute all subset sums as a set."""
        if not arr:
            return {0}
        sums = {0}
        for x in arr:
            sums = sums | {s + x for s in sums}
        return sums

    def find_min_replacement(arr, idx):
        """Find the smallest v < arr[idx] that is valid for position idx."""
        current_val = arr[idx]
        others = arr[:idx] + arr[idx+1:]
        others_set = set(others)
        
        # Compute subset sums of others as a set for O(1) lookup
        S_others = subset_sums_set(others)
        
        # We want the smallest v such that:
        # 1. v not in others_set (distinctness)
        # 2. (S_others + v) is disjoint from S_others
        
        for v in range(1, current_val):
            if v in others_set:
                continue
            
            # Check if any element in S_others + v is in S_others
            # Using set intersection is faster than np.isin for these sizes
            shifted = {s + v for s in S_others}
            if shifted & S_others:
                continue
            else:
                return v
                
        return None

    def is_valid(arr):
        """Check if all subset sums are distinct."""
        sums = subset_sums_set(arr)
        return len(sums) == (1 << len(arr))

    # Start with powers of 2 (known valid)
    arr = [1 << i for i in range(12)]
    
    # We want to minimize the max element.
    # Try to reduce elements, targeting largest first.
    
    best_arr = arr[:]
    best_max = max(arr)
    
    for _pass in range(200):
        if time.time() > deadline:
            break
        
        # Sort in descending order to target largest elements first
        arr.sort(reverse=True)
        
        improved = False
        for idx in range(len(arr)):
            if time.time() > deadline:
                break
            
            current_val = arr[idx]
            new_val = find_min_replacement(arr, idx)
            
            if new_val is not None and new_val < current_val:
                arr[idx] = new_val
                improved = True
        
        if not improved:
            break
        
        # Track best
        if max(arr) < best_max:
            best_max = max(arr)
            best_arr = arr[:]
    
    # Final verification
    if is_valid(best_arr):
        return best_arr
    return best_arr
# EVOLVE-BLOCK-END
