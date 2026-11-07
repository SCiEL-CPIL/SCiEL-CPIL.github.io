import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    N = 10_000_000
    
    # Compute sigma(n) for n=1..N using a more efficient sieve
    # Use int64 to be safe, though int32 should suffice for sigma(n) <= ~45M
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve: for i=1..N, add i to multiples of i
    # This is O(N log N) but with numpy vectorization it's fast enough
    for i in range(1, N + 1):
        if time.time() - start_time > DEADLINE - 10:
            break
        sigma[i::i] += i
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        return [11, 6]
    
    # Now group indices by sigma value
    # We want to find pairs (a,b) with sigma(a) = sigma(b) that minimize |a/b - 3/2|
    # This is equivalent to minimizing |2a - 3b| / (2b)
    
    # Strategy: For each sigma value, we have a set of numbers.
    # For each b in the set, we want a in the set closest to 1.5*b.
    # We can do this efficiently by sorting each group.
    
    # First, let's identify which sigma values appear more than once
    # and have a reasonable number of elements (to avoid huge groups)
    
    sig = sigma[1:N+1]
    
    # Use argsort to group by sigma value
    order = np.argsort(sig, kind='quicksort')
    sorted_sig = sig[order]
    sorted_idx = order + 1  # actual n values (1-indexed)
    
    # Find group boundaries
    changes = np.diff(sorted_sig) != 0
    boundaries = np.concatenate(([0], np.where(changes)[0] + 1, [N]))
    
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    num_groups = len(boundaries) - 1
    
    # Process groups in order of size (larger groups more likely to have good pairs)
    # But also consider that we want small error, so let's just process all groups
    # with a time check
    
    for gi in range(num_groups):
        if time.time() - start_time > DEADLINE - 3:
            break
            
        s = boundaries[gi]
        e = boundaries[gi+1]
        size = e - s
        
        if size <= 1:
            continue
        
        # Get the group's indices and sort them
        group = sorted_idx[s:e].copy()
        group.sort()
        
        # For each b in group, find a in group closest to 1.5*b
        # Vectorized: targets = 1.5 * group
        targets = (3 * group.astype(np.float64)) / 2.0
        pos = np.searchsorted(group, targets, side='left')
        
        size_int = len(group)
        
        # Candidate 1: group[pos] where pos < size
        mask1 = pos < size_int
        if np.any(mask1):
            p1 = pos[mask1]
            a_cand = group[p1]
            b_cand = group[mask1]
            # Exclude a == b
            valid = a_cand != b_cand
            if np.any(valid):
                a_v = a_cand[valid]
                b_v = b_cand[valid]
                # Compute error |a/b - 3/2| = |2a - 3b| / (2b)
                num = np.abs(2 * a_v.astype(np.int64) - 3 * b_v.astype(np.int64))
                den = 2 * b_v.astype(np.int64)
                errs = num.astype(np.float64) / den.astype(np.float64)
                min_i = np.argmin(errs)
                err_val = errs[min_i]
                if err_val < best_error:
                    best_error = err_val
                    best_a = int(a_v[min_i])
                    best_b = int(b_v[min_i])
        
        # Candidate 2: group[pos-1] where pos > 0
        mask2 = pos > 0
        if np.any(mask2):
            p2 = pos[mask2] - 1
            a_cand = group[p2]
            b_cand = group[mask2]
            valid = a_cand != b_cand
            if np.any(valid):
                a_v = a_cand[valid]
                b_v = b_cand[valid]
                num = np.abs(2 * a_v.astype(np.int64) - 3 * b_v.astype(np.int64))
                den = 2 * b_v.astype(np.int64)
                errs = num.astype(np.float64) / den.astype(np.float64)
                min_i = np.argmin(errs)
                err_val = errs[min_i]
                if err_val < best_error:
                    best_error = err_val
                    best_a = int(a_v[min_i])
                    best_b = int(b_v[min_i])
    
    # Ensure we return valid Python ints
    return [int(best_a), int(best_b)]
