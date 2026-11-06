"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 85  # Safety margin
    
    def subset_sums(arr):
        """Compute all subset sums using NumPy."""
        if not arr:
            return np.array([0], dtype=np.int64)
        sums = np.array([0], dtype=np.int64)
        for x in arr:
            sums = np.concatenate([sums, sums + x])
        return sums

    def find_min_replacement(arr, idx):
        """Find the smallest v < arr[idx] that is valid for position idx."""
        current_val = arr[idx]
        others = arr[:idx] + arr[idx+1:]
        others_set = set(others)
        
        # Compute subset sums of others
        S_others = subset_sums(others)
        
        # We want the smallest v such that:
        # 1. v not in others_set
        # 2. (S_others + v) is disjoint from S_others
        
        for v in range(1, current_val):
            if v in others_set:
                continue
            
            # Vectorized collision check
            # Check if any element in S_others + v is in S_others
            if np.any(np.isin(S_others + v, S_others)):
                continue
            else:
                return v
                
        return None

    # Start with powers of 2 (known valid)
    arr = [1 << i for i in range(12)]
    
    # We want to minimize the max element.
    # So we should try to reduce the largest elements.
    
    for _pass in range(100):
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
            
    return arr
# EVOLVE-BLOCK-END
