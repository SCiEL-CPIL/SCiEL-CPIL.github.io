import time
import numpy as np
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    denominators = list(range(2, 49))
    n = len(denominators)

    def lcm(a, b):
        return a * b // gcd(a, b)

    L = reduce(lcm, denominators)
    weights = [L // d for d in denominators]

    # Split items into two halves
    mid = n // 2
    half1 = weights[:mid]
    half2 = weights[mid:]

    def gen_feasible_sums_np(weights):
        sums = np.array([0], dtype=np.int64)
        masks = np.array([0], dtype=np.int64)
        
        for i, w in enumerate(weights):
            # Create new sums and masks
            new_sums = sums + w
            new_masks = masks | (1 << i)
            
            # Concatenate
            sums = np.concatenate([sums, new_sums])
            masks = np.concatenate([masks, new_masks])
            
            # Filter feasible sums (< L)
            valid = sums < L
            sums = sums[valid]
            masks = masks[valid]
            
            # Safety check for time
            if time.time() > deadline:
                break

        return sums, masks

    # Generate for half1
    sums1, masks1 = gen_feasible_sums_np(half1)
    
    if time.time() > deadline:
        # Fallback if generation took too long (shouldn't happen)
        return [0]

    # Generate for half2
    sums2, masks2 = gen_feasible_sums_np(half2)
    
    if time.time() > deadline:
        return [0]

    # Sort sums1 and sums2
    # We need to keep masks aligned
    idx1 = np.argsort(sums1)
    sums1_sorted = sums1[idx1]
    masks1_sorted = masks1[idx1]
    
    idx2 = np.argsort(sums2)
    sums2_sorted = sums2[idx2]
    masks2_sorted = masks2[idx2]

    # Two-pointer search
    # i points to sums1 (ascending)
    # j points to sums2 (ascending, start from end)
    i = 0
    j = len(sums2_sorted) - 1
    n1 = len(sums1_sorted)
    n2 = len(sums2_sorted)
    
    best_deficit = L
    best_mask1 = 0
    best_mask2 = 0
    
    while i < n1 and j >= 0:
        if time.time() > deadline:
            break
            
        s1 = sums1_sorted[i]
        s2 = sums2_sorted[j]
        total = s1 + s2
        
        if total < L:
            deficit = L - total
            if deficit < best_deficit:
                best_deficit = deficit
                best_mask1 = masks1_sorted[i]
                best_mask2 = masks2_sorted[j]
                
                if best_deficit == 1:
                    break
                
            # Try to increase sum by increasing s1
            i += 1
        else:
            # Sum is >= L, decrease s2
            j -= 1

    # Reconstruct indices
    final_indices = []
    for i in range(mid):
        if best_mask1 & (1 << i):
            final_indices.append(i)

    for i in range(len(half2)):
        if best_mask2 & (1 << i):
            final_indices.append(mid + i)

    # Fallback if no valid nonempty subset found
    if not final_indices:
        # Pick the largest single item
        max_idx = weights.index(max(weights))
        final_indices = [max_idx]

    return final_indices
