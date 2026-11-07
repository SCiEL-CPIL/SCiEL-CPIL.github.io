"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill
    
    N = 10000000
    
    # Compute sigma(n) for all n up to N using efficient sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    # For each i, add i to all multiples of i
    for i in range(1, N + 1):
        sigma[i::i] += i
    
    if time.time() > deadline - 15:
        return [11, 6]
    
    # Efficient grouping: use numpy to group by sigma value
    # Sort indices by sigma value, then find groups
    indices = np.arange(1, N + 1)
    sigma_vals = sigma[1:]
    
    # Sort by sigma value
    order = np.argsort(sigma_vals, kind='stable')
    sorted_sigma = sigma_vals[order]
    sorted_indices = indices[order]
    
    # Find group boundaries
    change_points = np.where(np.diff(sorted_sigma) != 0)[0] + 1
    group_starts = np.concatenate([[0], change_points])
    group_ends = np.concatenate([change_points, [len(sorted_indices)]])
    
    if time.time() > deadline - 10:
        return [11, 6]
    
    # Find groups with size >= 2
    group_sizes = group_ends - group_starts
    valid_groups = np.where(group_sizes >= 2)[0]
    
    if len(valid_groups) == 0:
        return [11, 6]
    
    # Sort valid groups by size descending (larger groups more likely to have good pairs)
    valid_groups = valid_groups[np.argsort(-group_sizes[valid_groups])]
    
    best_error = 1.0
    best_pair = (11, 6)
    
    for gi in valid_groups:
        if time.time() > deadline - 3:
            break
        
        s_start = group_starts[gi]
        s_end = group_ends[gi]
        nums_sorted = sorted_indices[s_start:s_end].astype(np.int64)
        n = len(nums_sorted)
        
        if n < 2:
            continue
        
        # For each b, find a closest to 3b/2 using binary search
        # Use vectorized approach where possible
        targets = (3 * nums_sorted) // 2
        
        # For each target, find insertion position
        # We need to check elements at pos and pos-1 for each b
        # Do this in batches to be efficient
        
        for i in range(n):
            if time.time() > deadline - 2:
                break
            b = int(nums_sorted[i])
            target_a = int(targets[i])
            pos = bisect_left(nums_sorted, target_a)
            
            candidates = []
            if pos < n:
                candidates.append(int(nums_sorted[pos]))
            if pos > 0:
                candidates.append(int(nums_sorted[pos - 1]))
            
            for a in candidates:
                if a == b:
                    continue
                error_num = abs(2 * a - 3 * b)
                error_den = 2 * b
                error = error_num / error_den
                if error < best_error:
                    best_error = error
                    best_pair = (a, b)
    
    a, b = best_pair
    a = int(a)
    b = int(b)
    if sigma[a] != sigma[b] or a == b:
        return [11, 6]
    return [a, b]

# EVOLVE-BLOCK-END
