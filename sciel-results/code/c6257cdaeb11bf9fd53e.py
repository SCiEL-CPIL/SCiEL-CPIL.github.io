import numpy as np
import time

P = {'maximum': 10000000, 'target': [3, 2]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    N = 10000000

    # Compute sigma(n) for all n up to N using sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        if time.time() > deadline - 30.0:
             # If sieve is too slow, return fallback
             # But we need sigma to be at least partially done?
             # No, if we break, sigma is incomplete.
             # Better to let it run or use a faster method.
             # For now, assume it finishes.
             pass
        sigma[i::i] += i

    if time.time() > deadline - 10.0:
        return [11, 6]

    # Build dictionary: sigma value -> list of n with that sigma
    # Sort indices by sigma value
    indices = np.arange(1, N + 1, dtype=np.int64)
    # Fix: argsort on sigma[1:] gives indices 0..N-1.
    # We map these to n values 1..N using indices array.
    sorted_indices = indices[np.argsort(sigma[1:], kind='mergesort')]
    
    # Fix: Use sigma[sorted_indices] because sorted_indices contains n values (1..N)
    # and sigma is indexed by n (0..N).
    sorted_sigma = sigma[sorted_indices]
    
    if time.time() > deadline - 5.0:
        return [11, 6]
    
    # Find groups where sigma value is the same
    unique_sigma, start_idx = np.unique(sorted_sigma, return_index=True)
    
    best_value = 1.0 / 3.0  # from [11, 6]
    best_pair = (11, 6)
    
    # Get group sizes
    end_idx = np.empty(len(unique_sigma), dtype=np.int64)
    end_idx[:-1] = start_idx[1:]
    end_idx[-1] = N
    group_sizes = end_idx - start_idx
    
    # Sort groups by size descending, process largest first
    group_order = np.argsort(-group_sizes)
    
    max_groups_to_process = len(group_order)
    
    for gi in group_order[:max_groups_to_process]:
        if time.time() > deadline - 2.0:
            break
        if group_sizes[gi] < 2:
            break  # sorted descending, so all remaining are < 2
        
        s_start = start_idx[gi]
        s_end = end_idx[gi]
        group_vals = sorted_indices[s_start:s_end]
        
        if len(group_vals) < 2:
            continue
            
        # Vectorized search for best pair in this group
        # For each b in group_vals, find a in group_vals minimizing |a/b - 1.5|
        # Equivalent to minimizing |2a - 3b| / (2b)
        
        targets = 1.5 * group_vals
        pos = np.searchsorted(group_vals, targets)
        
        # Candidate 1: group_vals[pos]
        # Candidate 2: group_vals[pos-1]
        
        # Handle bounds for pos
        pos1 = pos
        pos2 = pos - 1
        
        valid1 = (pos1 >= 0) & (pos1 < len(group_vals))
        valid2 = (pos2 >= 0) & (pos2 < len(group_vals))
        
        # We need to compute error for valid candidates where a != b
        
        # Candidate 1
        if np.any(valid1):
            a1 = group_vals[pos1[valid1]]
            b1 = group_vals[valid1]
            # Exclude a == b
            mask1 = a1 != b1
            if np.any(mask1):
                a1 = a1[mask1]
                b1 = b1[mask1]
                err1 = np.abs(2*a1 - 3*b1) / (2.0 * b1)
                min_idx1 = np.argmin(err1)
                val1 = err1[min_idx1]
                if val1 < best_value:
                    best_value = val1
                    best_pair = (int(a1[min_idx1]), int(b1[min_idx1]))

        # Candidate 2
        if np.any(valid2):
            a2 = group_vals[pos2[valid2]]
            b2 = group_vals[valid2]
            # Exclude a == b
            mask2 = a2 != b2
            if np.any(mask2):
                a2 = a2[mask2]
                b2 = b2[mask2]
                err2 = np.abs(2*a2 - 3*b2) / (2.0 * b2)
                min_idx2 = np.argmin(err2)
                val2 = err2[min_idx2]
                if val2 < best_value:
                    best_value = val2
                    best_pair = (int(a2[min_idx2]), int(b2[min_idx2]))

    return [int(best_pair[0]), int(best_pair[1])]
