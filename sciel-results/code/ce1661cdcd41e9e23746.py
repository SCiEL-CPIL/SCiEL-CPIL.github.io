import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin
    MAX_N = 10000000
    
    # 1. Compute sigma(n) for n=1..MAX_N
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    # Sieve of Eratosthenes style for divisor sum
    # sigma[n] = sum of divisors of n
    # For each d, add d to all multiples of d
    for d in range(1, MAX_N + 1):
        sigma[d::d] += d
        
    if time.time() > deadline:
        return [11, 6]

    # 2. Sort indices by sigma value
    indices = np.arange(1, MAX_N + 1, dtype=np.int64)
    
    # Stable sort to preserve original order (1..N) within equal sigma values
    order = np.argsort(sigma[1:], kind='stable')
    
    sorted_sigma = sigma[1:][order]
    sorted_n = indices[order]
    
    if time.time() > deadline:
        return [11, 6]

    # 3. Identify blocks of equal sigma
    changes = np.where(sorted_sigma[:-1] != sorted_sigma[1:])[0] + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [len(sorted_n)]))
    
    # Filter blocks with size >= 2
    valid_block_indices = np.where(ends - starts >= 2)[0]
    
    best_a = 11
    best_b = 6
    # Error = |a/b - 3/2| = |2a - 3b| / (2b)
    # For 11, 6: |22 - 18| / 12 = 4/12 = 1/3.
    best_err_num = 1
    best_err_den = 3
    
    # 4. Search for best pair in each valid block
    for idx in valid_block_indices:
        if time.time() > deadline:
            break
            
        s = starts[idx]
        e = ends[idx]
        block_n = sorted_n[s:e]
        k = e - s
        
        # Vectorized search for a closest to 1.5 * b
        targets = 1.5 * block_n
        pos = np.searchsorted(block_n, targets, side='left')
        
        # Candidate indices: pos-1, pos, pos+1
        # Clip to valid range [0, k-1]
        cand_idx_1 = np.clip(pos - 1, 0, k - 1)
        cand_idx_2 = np.clip(pos, 0, k - 1)
        cand_idx_3 = np.clip(pos + 1, 0, k - 1)
        
        # Get candidate a values
        a1 = block_n[cand_idx_1]
        a2 = block_n[cand_idx_2]
        a3 = block_n[cand_idx_3]
        
        # Compute errors |2a - 3b|
        # b is block_n
        err1 = np.abs(2 * a1 - 3 * block_n)
        err2 = np.abs(2 * a2 - 3 * block_n)
        err3 = np.abs(2 * a3 - 3 * block_n)
        
        # Combine all candidates
        # We need to find the minimum error, and if 0, the next minimum.
        # To avoid large memory allocations, we can check min of each and then handle 0s.
        
        # First, find global min of the three arrays
        min_err_1 = np.min(err1)
        min_err_2 = np.min(err2)
        min_err_3 = np.min(err3)
        
        global_min = min(min_err_1, min_err_2, min_err_3)
        
        if global_min == 0:
            # Need to find the smallest non-zero error
            # We can concatenate and filter, or just find min non-zero in each
            # Since k is small, concatenation is fine.
            all_errs = np.concatenate([err1, err2, err3])
            non_zero_mask = all_errs > 0
            if not np.any(non_zero_mask):
                # All pairs have error 0? Impossible unless a/b=1.5 for all?
                # If a/b=1.5, then 2a=3b.
                # If all pairs in block satisfy this, then all a=1.5b.
                # Since a,b are integers, b must be even, a=3k, b=2k.
                # If block contains only such pairs, we can't pick a valid pair.
                # But we need a pair from the block.
                # If all pairs are invalid, we skip this block.
                continue
                
            min_non_zero = np.min(all_errs[non_zero_mask])
            
            # Find which candidate this came from
            # We need the corresponding b
            # To do this efficiently without complex mapping, we can just 
            # re-iterate or use argmax on the masked array?
            # Actually, we just need ONE pair with this error.
            # Let's find the index in the concatenated array.
            # But we also need to map back to a and b.
            
            # Simpler: just find min non-zero in each err array and pick the best
            min_nz_1 = np.min(err1[err1 > 0]) if np.any(err1 > 0) else np.inf
            min_nz_2 = np.min(err2[err2 > 0]) if np.any(err2 > 0) else np.inf
            min_nz_3 = np.min(err3[err3 > 0]) if np.any(err3 > 0) else np.inf
            
            best_local_err = min(min_nz_1, min_nz_2, min_nz_3)
            
            if best_local_err == np.inf:
                continue
                
            # Find the pair
            if best_local_err == min_nz_1:
                # Find index of min in err1
                # There might be multiple, pick the first
                local_idx = np.argmin(np.where(err1 > 0, err1, np.inf))
                a = a1[local_idx]
                b = block_n[local_idx]
            elif best_local_err == min_nz_2:
                local_idx = np.argmin(np.where(err2 > 0, err2, np.inf))
                a = a2[local_idx]
                b = block_n[local_idx]
            else:
                local_idx = np.argmin(np.where(err3 > 0, err3, np.inf))
                a = a3[local_idx]
                b = block_n[local_idx]
                
            num = best_local_err
            den = 2 * b
        else:
            # Global min is non-zero
            if global_min == min_err_1:
                local_idx = np.argmin(err1)
                a = a1[local_idx]
                b = block_n[local_idx]
            elif global_min == min_err_2:
                local_idx = np.argmin(err2)
                a = a2[local_idx]
                b = block_n[local_idx]
            else:
                local_idx = np.argmin(err3)
                a = a3[local_idx]
                b = block_n[local_idx]
                
            num = global_min
            den = 2 * b
            
        # Update global best
        # Compare num/den with best_err_num/best_err_den
        # num / den < best_num / best_den  => num * best_den < best_num * den
        if num * best_err_den < best_err_num * den:
            best_a = int(a)
            best_b = int(b)
            best_err_num = num
            best_err_den = den

    # Final check
    if best_err_num == 0:
        return [11, 6]
        
    return [best_a, best_b]
