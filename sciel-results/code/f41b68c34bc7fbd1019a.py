"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import sys
from fractions import Fraction

MAX_N = 10_000_000

def construct():
    t0 = time.time()
    deadline = t0 + 85  # safety margin before 90s limit

    # Sieve for sigma(n) using numpy for speed
    try:
        import numpy as np
        MAX_N = 10_000_000
        sigma_arr = np.zeros(MAX_N + 1, dtype=np.int64)
        for i in range(1, MAX_N + 1):
            sigma_arr[i::i] += i
        
        # Check time
        if time.time() > deadline - 10:
            return [11, 6]
        
        # Sort by sigma value
        sigma_vals = sigma_arr[1:]
        order = np.argsort(sigma_vals, kind='stable')
        sorted_sigma = sigma_vals[order]
        sorted_n = (order + 1).astype(np.int32)
        
        # Find groups of same sigma
        best_err = float('inf')
        best_pair = (11, 6)
        
        # Find group boundaries using vectorized comparison
        diff = np.diff(sorted_sigma)
        boundaries = np.where(diff != 0)[0] + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [len(sorted_n)]])
        
        n_groups = len(starts)
        
        for g in range(n_groups):
            s, e = starts[g], ends[g]
            group_size = e - s
            if group_size < 2:
                continue
            
            group = sorted_n[s:e]
            
            # For each b in group, target a = 3b/2
            # Vectorized: compute targets and find closest in group
            b_vals = group.astype(np.float64)
            targets = 1.5 * b_vals
            
            # Binary search for each target in the sorted group
            # np.searchsorted gives insertion points
            insert_pos = np.searchsorted(group, targets, side='left')
            
            # Check candidate positions: insert_pos and insert_pos - 1
            pos1 = np.clip(insert_pos, 0, group_size - 1)
            pos0 = np.clip(insert_pos - 1, 0, group_size - 1)
            
            a_candidates1 = group[pos1].astype(np.float64)
            a_candidates0 = group[pos0].astype(np.float64)
            
            # Error for each candidate
            err1 = np.abs(2 * a_candidates1 - 3 * b_vals) / (2 * b_vals)
            err0 = np.abs(2 * a_candidates0 - 3 * b_vals) / (2 * b_vals)
            
            # Exclude a == b cases
            same1 = (a_candidates1 == b_vals)
            same0 = (a_candidates0 == b_vals)
            err1[same1] = np.inf
            err0[same0] = np.inf
            
            combined = np.minimum(err1, err0)
            min_idx = np.argmin(combined)
            min_err = combined[min_idx]
            
            if min_err < best_err and min_err > 0:
                best_err = float(min_err)
                # Determine which candidate was better
                if err1[min_idx] <= err0[min_idx]:
                    a_val = int(group[pos1[min_idx]])
                else:
                    a_val = int(group[pos0[min_idx]])
                b_val = int(b_vals[min_idx])
                best_pair = (a_val, b_val)
            
            # Early termination if error is very small
            if best_err < 1e-10:
                break
            
            # Time check
            if time.time() > deadline - 5:
                break
        
        if best_pair != (11, 6):
            return [int(best_pair[0]), int(best_pair[1])]
        return [11, 6]
        
    except ImportError:
        # Pure Python fallback
        MAX_N = 10_000_000
        sigma = [0] * (MAX_N + 1)
        for i in range(1, MAX_N + 1):
            for j in range(i, MAX_N + 1, i):
                sigma[j] += i
        
        if time.time() > deadline - 5:
            return [11, 6]
        
        pairs = [(sigma[n], n) for n in range(1, MAX_N + 1)]
        pairs.sort()
        
        best_err = float('inf')
        best_pair = (11, 6)
        
        i = 0
        n_total = len(pairs)
        while i < n_total:
            j = i + 1
            while j < n_total and pairs[j][0] == pairs[i][0]:
                j += 1
            group_size = j - i
            if group_size >= 2:
                group = [pairs[k][1] for k in range(i, j)]
                for idx_b in range(len(group)):
                    b_val = group[idx_b]
                    target_a = 3 * b_val / 2
                    lo, hi = 0, len(group) - 1
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if group[mid] < target_a:
                            lo = mid + 1
                        else:
                            hi = mid
                    candidates = []
                    if lo < len(group):
                        candidates.append(group[lo])
                    if lo - 1 >= 0:
                        candidates.append(group[lo - 1])
                    for a_val in candidates:
                        if a_val != b_val:
                            err_num = abs(2 * a_val - 3 * b_val)
                            err_den = 2 * b_val
                            err = err_num / err_den
                            if err < best_err and err > 0:
                                best_err = err
                                best_pair = (a_val, b_val)
            i = j
            if time.time() > deadline - 3:
                break
        
        if best_pair != (11, 6):
            return [int(best_pair[0]), int(best_pair[1])]
        return [11, 6]

    # Group numbers by sigma value
    # We only need sigma values that appear at least twice
    # To save memory, use a dict mapping sigma -> list of numbers
    # But that's expensive. Instead, sort by sigma.
    
    # Create list of (sigma[n], n) pairs
    # This is 10M entries - use arrays
    import array
    # Actually, let's just sort indices by sigma value
    # sigma is a list of 10M+1 ints
    
    # Better approach: for each sigma value, collect all n with that sigma
    # Use a dictionary but only for sigma values that repeat
    # First pass: count occurrences
    # This is memory intensive. Let's use a different approach.
    
    # For each n, sigma[n] is known. We want to find pairs with same sigma.
    # Sort n by sigma[n], then adjacent elements in the sorted order have same sigma.
    
    # Create list of indices sorted by sigma value
    # 10M sort might be slow in pure Python...
    
    # Alternative: use numpy if available
    # This block is replaced by the new numpy-based approach above
    pass

# EVOLVE-BLOCK-END
