import numpy as np
import math
import time

def construct():
    start_time = time.time()
    N = 30
    # Calculate LCM
    L = 1
    for i in range(1, N + 1):
        L = L * i // math.gcd(L, i)
    
    weights = np.array([L // i for i in range(1, N + 1)], dtype=np.int64)
    
    # Split into two halves
    mid = N // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    def generate_sums(w):
        sums = np.array([0], dtype=np.int64)
        for weight in w:
            sums = np.concatenate([sums - weight, sums, sums + weight])
        return sums
    
    sums1 = generate_sums(w1)
    sums2 = generate_sums(w2)
    
    # Sort sums1 for binary search
    sums1.sort()
    
    # Vectorized search for best pair
    # We want a + b close to 0
    # b is from sums2
    # target is -b
    targets = -sums2
    indices = np.searchsorted(sums1, targets)
    
    # Handle boundaries
    # indices can be 0 to len(sums1)
    # If indices == len(sums1), the closest is the last element.
    # If indices == 0, the closest is the first element.
    
    # We check indices and indices-1
    # Create arrays of indices to check
    # To avoid out of bounds, we can pad or clip
    
    # Let's just create candidate arrays
    # Left candidates: sums1[indices-1] + sums2
    # Right candidates: sums1[indices] + sums2
    
    # We need to be careful with indices=0 and indices=len
    # np.clip handles this
    
    idx_left = np.clip(indices - 1, 0, len(sums1) - 1)
    idx_right = np.clip(indices, 0, len(sums1) - 1)
    
    cand_left = sums1[idx_left] + sums2
    cand_right = sums1[idx_right] + sums2
    
    all_cands = np.concatenate([cand_left, cand_right])
    
    # Find min non-zero absolute value
    abs_cands = np.abs(all_cands)
    # Mask out zeros
    abs_cands[all_cands == 0] = np.iinfo(np.int64).max
    
    min_val = np.min(abs_cands)
    min_idx = np.argmin(abs_cands)
    
    # Determine which candidate it was
    if min_idx < len(sums2):
        b_idx = min_idx
        a_idx = idx_left[b_idx]
        a_val = sums1[a_idx]
        b_val = sums2[b_idx]
    else:
        b_idx = min_idx - len(sums2)
        a_idx = idx_right[b_idx]
        a_val = sums1[a_idx]
        b_val = sums2[b_idx]
        
    # Reconstruct coefficients
    def find_coeffs(target, w):
        # Recursive search
        # w is numpy array
        # target is int
        coeffs = [0] * len(w)
        
        def backtrack(i, current_sum):
            if i == len(w):
                return current_sum == target
            # Try -1, 0, 1
            for c in [-1, 0, 1]:
                coeffs[i] = c
                if backtrack(i + 1, current_sum + c * w[i]):
                    return True
            return False
        
        if backtrack(0, 0):
            return coeffs
        else:
            # Should not happen
            return [0] * len(w)
            
    c1 = find_coeffs(a_val, w1)
    c2 = find_coeffs(b_val, w2)
    
    return c1 + c2
