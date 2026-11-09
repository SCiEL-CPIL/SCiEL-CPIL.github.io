"""Program to find a,b with sigma(a)=sigma(b) minimizing |a/b - 3/2|, where error must be > 0."""
import math
import time
import numpy as np
from fractions import Fraction

def construct():
    t_start = time.time()
    deadline = t_start + 82  # safety margin before 90s kill
    
    N = 10_000_000
    
    # 1. Compute sigma(n) for n in [1, N] using NumPy sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
    
    if time.time() > deadline:
        return [11, 6]
    
    # 2. Sort indices by sigma value
    sig_vals = sigma[1:]  # shape (N,)
    sort_idx = np.argsort(sig_vals, kind='stable')
    sorted_sig = sig_vals[sort_idx]
    sorted_nums = np.arange(1, N + 1, dtype=np.int64)[sort_idx]
    
    # Find group boundaries
    changes = np.concatenate(([True], sorted_sig[1:] != sorted_sig[:-1]))
    group_starts = np.where(changes)[0]
    group_ends = np.concatenate((group_starts[1:], [N]))
    
    # 3. Search within each group for best pair
    # We want to minimize |a/b - 3/2| = |2a - 3b| / (2b)
    # For exact comparison, use integer arithmetic: minimize |2a - 3b| * best_den < best_num * 2 * b
    
    best_a = 11
    best_b = 6
    # Error: |11/6 - 3/2| = |22/12 - 18/12| = 4/12 = 1/3
    # Represented as best_num / best_den where best_num = |2a - 3b|, best_den = 2b
    best_num = abs(2 * 11 - 3 * 6)  # = 4
    best_den = 2 * 6  # = 12
    
    for gi in range(len(group_starts)):
        if time.time() > deadline - 3:
            break
        
        gs = int(group_starts[gi])
        ge = int(group_ends[gi])
        group_size = ge - gs
        
        if group_size < 2:
            continue
        
        # Extract and sort the numbers in this group
        group_arr = np.sort(sorted_nums[gs:ge])
        k = len(group_arr)
        
        if k < 2:
            continue
        
        # For each a in group_arr, find b in group_arr close to 2a/3
        # We want to minimize |2a - 3b| / (2b)
        # For fixed a, optimal b is near 2a/3
        
        # Vectorized approach:
        targets = (2.0 * group_arr.astype(np.float64)) / 3.0
        idxs = np.searchsorted(group_arr, targets, side='left')
        
        # Clamp indices
        idxs = np.clip(idxs, 0, k - 1)
        
        # Candidate 1: group_arr[idxs]
        # Candidate 2: group_arr[idxs - 1] (if idxs > 0)
        cand1 = group_arr[idxs]
        cand2 = group_arr[np.clip(idxs - 1, 0, k - 1)]
        
        # Compute |2a - 3b| for both candidates
        # group_arr is the 'a' values, cand1/cand2 are the 'b' values
        diff1 = np.abs(2 * group_arr - 3 * cand1).astype(np.int64)
        diff2 = np.abs(2 * group_arr - 3 * cand2).astype(np.int64)
        
        # We want to minimize diff / (2*b)
        # For exact comparison with current best (best_num/best_den):
        # diff / (2*b) < best_num / best_den
        # diff * best_den < best_num * 2 * b
        
        # Avoid division by zero (cand should be >= 1, which it is since we start from 1)
        # score1 = diff1 / (2*cand1), score2 = diff2 / (2*cand2)
        # But we compare exactly using integers
        
        # For each position i, check both candidates
        # Candidate 1 is better if: diff1[i] * best_den < best_num * 2 * cand1[i]
        # Candidate 2 is better if: diff2[i] * best_den < best_num * 2 * cand2[i]
        
        # Find the best among all positions and both candidates
        # We'll compute the "score" as a float for argmin, then verify exactly
        
        # Use float scores for finding candidates, then exact check
        with np.errstate(divide='ignore', invalid='ignore'):
            score1 = diff1.astype(np.float64) / (2.0 * cand1.astype(np.float64))
            score2 = diff2.astype(np.float64) / (2.0 * cand2.astype(np.float64))
        
        # Replace any non-finite scores with infinity
        score1 = np.where(np.isfinite(score1), score1, np.inf)
        score2 = np.where(np.isfinite(score2), score2, np.inf)
        
        all_scores = np.minimum(score1, score2)
        
        # Skip if all scores are infinity
        if np.all(np.isinf(all_scores)):
            continue
        
        best_local_idx = int(np.argmin(all_scores))
        
        # Determine which candidate was better at this index
        if score1[best_local_idx] <= score2[best_local_idx]:
            a_val = int(group_arr[best_local_idx])
            b_val = int(cand1[best_local_idx])
            diff_val = int(diff1[best_local_idx])
        else:
            a_val = int(group_arr[best_local_idx])
            b_val = int(cand2[best_local_idx])
            diff_val = int(diff2[best_local_idx])
        
        # Exact comparison: diff_val / (2 * b_val) < best_num / best_den
        # diff_val * best_den < best_num * 2 * b_val
        lhs = diff_val * best_den
        rhs = best_num * 2 * b_val
        
        if lhs < rhs:
            best_num = diff_val
            best_den = 2 * b_val
            best_a = a_val
            best_b = b_val
    
    # Ensure a != b and error > 0
    # If a == b, error is |1 - 3/2| = 1/2, which is worse than 1/3 (our initial best)
    # But let's verify our result is valid
    if best_a == best_b or best_num == 0:
        # Fallback
        best_a = 11
        best_b = 6
    
    # Verify: sigma(best_a) should equal sigma(best_b)
    # This should be guaranteed by construction, but let's be safe
    if sigma[best_a] != sigma[best_b]:
        best_a = 11
        best_b = 6
    
    return [int(best_a), int(best_b)]
