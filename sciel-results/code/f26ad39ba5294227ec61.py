import time
import math
import numpy as np
from bisect import bisect_left

def construct():
    t_start = time.time()
    deadline = t_start + 85.0  # Safety margin before 90s kill
    
    N = 10_000_000
    
    # Step 1: Compute sigma(n) for all n in [1, N] using NumPy sieve
    # sigma[n] = sum of all divisors of n
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
    
    if time.time() > deadline - 5:
        return [11, 6]
    
    # Step 2: Sort indices by sigma value, then by index
    # This groups numbers with the same sigma value together
    indices = np.arange(1, N + 1)
    keys = sigma[indices]
    
    # np.lexsort sorts by the last key first
    # So (indices, keys) sorts by keys (sigma values), then by indices
    order = np.lexsort((indices, keys))
    sorted_n = indices[order]
    sorted_s = keys[order]
    
    if time.time() > deadline - 3:
        return [11, 6]
    
    # Step 3: Build mapping from sigma value to (start, end) range in sorted arrays
    # This avoids building huge lists in Python
    sigma_ranges = []  # list of (sigma_val, start_idx, end_idx)
    current_s = sorted_s[0]
    start_idx = 0
    
    for i in range(1, len(sorted_s)):
        if sorted_s[i] != current_s:
            if i - start_idx >= 2:  # Only keep groups with at least 2 elements
                sigma_ranges.append((int(current_s), start_idx, i))
            current_s = sorted_s[i]
            start_idx = i
    # Last group
    if len(sorted_s) - start_idx >= 2:
        sigma_ranges.append((int(current_s), start_idx, len(sorted_s)))
    
    if time.time() > deadline - 2:
        return [11, 6]
    
    # Step 4: For each group, find the pair minimizing |a/b - 3/2|
    best_a, best_b = 11, 6
    best_err = abs(11/6 - 3/2)  # = 1/3 ≈ 0.3333
    
    # Convert sorted_n to a Python list for faster bisect operations
    # Actually, let's keep it as numpy array and use numpy searchsorted
    # But bisect on a list might be faster for small groups
    
    # Strategy: For each group, for each b in the group, find a closest to 3b/2
    # Use binary search within the group's slice
    
    for s_val, lo, hi in sigma_ranges:
        if time.time() > deadline - 1:
            break
        
        group_size = hi - lo
        if group_size < 2:
            continue
        
        # Get the slice of n values for this group
        # sorted_n[lo:hi] is already sorted
        group = sorted_n[lo:hi]
        
        # For each b in group, find a in group closest to 3b/2
        # Use numpy searchsorted for efficiency
        targets = (3 * group.astype(np.float64)) / 2.0
        
        # Find insertion points
        pos = np.searchsorted(group, targets, side='left')
        
        # Check pos and pos-1 for each element
        # Vectorize: for each i, check group[pos[i]] and group[pos[i]-1]
        
        # pos[i] is the index in group where target[i] would be inserted
        # Check pos[i] (if in bounds) and pos[i]-1 (if in bounds)
        
        # Create safe indices
        pos_safe = pos.copy()
        pos_safe[pos_safe >= group_size] = group_size - 1
        pos_safe[pos_safe < 0] = 0
        
        # Check candidate at pos
        cand_a = group[pos_safe]
        cand_b = group
        # Exclude cases where a == b
        valid_mask = cand_a != cand_b
        if np.any(valid_mask):
            errs = np.abs(cand_a.astype(np.float64) / cand_b.astype(np.float64) - 1.5)
            errs[~valid_mask] = np.inf
            min_idx = np.argmin(errs)
            if errs[min_idx] < best_err and errs[min_idx] > 0:
                best_err = float(errs[min_idx])
                best_a = int(cand_a[min_idx])
                best_b = int(cand_b[min_idx])
        
        # Also check pos-1
        pos_minus = pos - 1
        pos_minus_safe = pos_minus.copy()
        pos_minus_safe[pos_minus_safe < 0] = 0
        pos_minus_safe[pos_minus_safe >= group_size] = group_size - 1
        
        cand_a2 = group[pos_minus_safe]
        valid_mask2 = cand_a2 != cand_b
        if np.any(valid_mask2):
            errs2 = np.abs(cand_a2.astype(np.float64) / cand_b.astype(np.float64) - 1.5)
            errs2[~valid_mask2] = np.inf
            min_idx2 = np.argmin(errs2)
            if errs2[min_idx2] < best_err and errs2[min_idx2] > 0:
                best_err = float(errs2[min_idx2])
                best_a = int(cand_a2[min_idx2])
                best_b = int(cand_b[min_idx2])
    
    # Final validation: verify sigma(best_a) == sigma(best_b) and best_a != best_b
    if best_a == best_b or int(sigma[best_a]) != int(sigma[best_b]):
        # Fallback to a known valid pair
        return [11, 6]
    
    # Ensure types are Python int
    return [int(best_a), int(best_b)]
