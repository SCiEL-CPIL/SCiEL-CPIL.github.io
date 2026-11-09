import time
import numpy as np
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    N = 10_000_000
    
    # 1. Compute sigma(n) for n in [1, N] using NumPy sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
    
    # 2. Group numbers by sigma value
    sig_vals = sigma[1:]  # shape (N,)
    
    # Sort indices by sigma value
    sort_idx = np.argsort(sig_vals, kind='stable')
    sorted_sig = sig_vals[sort_idx]
    sorted_nums = np.arange(1, N + 1, dtype=np.int64)[sort_idx]
    
    # Find group boundaries
    changes = np.concatenate(([True], sorted_sig[1:] != sorted_sig[:-1]))
    group_starts = np.where(changes)[0]
    group_ends = np.concatenate((group_starts[1:], [N]))
    
    # 3. Sort groups by size descending (larger groups first)
    group_sizes = group_ends - group_starts
    # Only consider groups with size >= 2
    valid_mask = group_sizes >= 2
    valid_starts = group_starts[valid_mask]
    valid_ends = group_ends[valid_mask]
    valid_sizes = group_sizes[valid_mask]
    
    # Sort by size descending
    size_order = np.argsort(valid_sizes)[::-1]
    
    # 4. Search within each group for best pair
    best_a = 11
    best_b = 6
    # Error: |11/6 - 3/2| = |22/12 - 18/12| = 4/12 = 1/3
    best_num = 4  # numerator of |2a - 3b|
    best_den = 12  # denominator = 2b
    
    for gi in range(len(valid_starts)):
        if time.time() > deadline:
            break
        
        gs = valid_starts[size_order[gi]]
        ge = valid_ends[size_order[gi]]
        
        # Extract the sorted numbers in this group
        group_arr = np.sort(sorted_nums[gs:ge])
        group_len = len(group_arr)
        
        if group_len < 2:
            continue
        
        # For each a in group_arr, find b in group_arr close to 2a/3
        # We want to minimize |2a - 3b| / (2b)
        # For fixed a, optimal b is near 2a/3
        
        # Vectorized approach:
        # targets = 2*group_arr / 3
        # Use searchsorted to find insertion points
        targets = (2.0 * group_arr) / 3.0
        idxs = np.searchsorted(group_arr, targets, side='left')
        
        # Clamp indices
        idxs = np.clip(idxs, 0, group_len - 1)
        
        # Candidate 1: group_arr[idxs]
        # Candidate 2: group_arr[idxs - 1] (if idxs > 0)
        cand1 = group_arr[idxs]
        cand2 = group_arr[np.clip(idxs - 1, 0, group_len - 1)]
        
        # Compute |2a - 3b| for both candidates
        diff1 = np.abs(2 * group_arr - 3 * cand1)
        diff2 = np.abs(2 * group_arr - 3 * cand2)
        
        # We want to minimize diff / (2*b) = diff / (2*cand)
        # So score1 = diff1 / (2*cand1), score2 = diff2 / (2*cand2)
        # Avoid division by zero (cand should be >= 1)
        score1 = diff1 / (2.0 * cand1)
        score2 = diff2 / (2.0 * cand2)
        
        # Find the best overall
        # Combine into one array
        all_scores = np.minimum(score1, score2)
        best_local_idx = np.argmin(all_scores)
        
        # Determine which candidate was better
        if score1[best_local_idx] <= score2[best_local_idx]:
            best_b_local = cand1[best_local_idx]
            best_diff = diff1[best_local_idx]
        else:
            best_b_local = cand2[best_local_idx]
            best_diff = diff2[best_local_idx]
        
        best_a_local = group_arr[best_local_idx]
        
        # Compare with global best using exact integer arithmetic
        # best_diff / (2 * best_b_local) vs best_num / best_den
        # best_diff * best_den vs best_num * 2 * best_b_local
        lhs = int(best_diff) * best_den
        rhs = best_num * int(2 * best_b_local)
        
        if lhs < rhs:
            best_num = int(best_diff)
            best_den = int(2 * best_b_local)
            best_a = int(best_a_local)
            best_b = int(best_b_local)
    
    # Ensure a != b (if a == b, error is |1 - 3/2| = 1/2, which is worse than 1/3)
    # But let's check: if best_a == best_b, we need to find another pair
    if best_a == best_b:
        # Find next best pair
        # This shouldn't happen often since we're looking for ratio near 3/2
        # But just in case
        best_a = 11
        best_b = 6
        best_num = 4
        best_den = 12
    
    return [int(best_a), int(best_b)]
