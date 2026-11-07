import time
import math
import bisect
import numpy as np

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin for 60s limit (or 90s if specified, but 55s is safe)
    
    N = 10_000_000
    
    # 1. Compute sigma(n) using NumPy
    # sigma[n] = sum of divisors of n
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve for sigma
    # This is O(N log N) operations
    for i in range(1, N + 1):
        if time.time() - start_time > deadline - 5.0:
            # If sigma computation takes too long, return a fallback
            # Fallback: 11, 6 (sigma(11)=12, sigma(6)=12). Ratio 11/6 = 1.833. Error |1.833-1.5|=0.333. Score 3.
            # We can do better with a quick search if we have time, but let's just return a valid pair.
            # Actually, 1 and 2? sigma(1)=1, sigma(2)=3. No.
            # 3 and 4? sigma(3)=4, sigma(4)=7. No.
            # 2 and 3? sigma(2)=3, sigma(3)=4. No.
            # 1 and 3? sigma(1)=1, sigma(3)=4. No.
            # Known pair: 6 and 11.
            return [11, 6]
        sigma[i::i] += i
        
    if time.time() - start_time > deadline:
        return [11, 6]

    # 2. Group by sigma value
    vals = sigma[1:]
    
    # Sort indices by sigma value
    sorted_indices = np.argsort(vals)
    sorted_vals = vals[sorted_indices]
    
    # Find group boundaries
    # diff is 0 where consecutive values are equal
    diff = np.diff(sorted_vals)
    changes = np.where(diff != 0)[0]
    
    # starts: index in sorted array where a new group begins
    starts = np.concatenate(([0], changes + 1))
    # ends: index in sorted array where a group ends (exclusive)
    ends = np.concatenate((changes + 1, [N]))
    
    best_error = float('inf')
    best_a = 11
    best_b = 6
    best_num = abs(2 * 11 - 3 * 6) # 4
    best_den = 6
    
    # Iterate through groups
    for i in range(len(starts)):
        if time.time() - start_time > deadline:
            break
            
        s = starts[i]
        e = ends[i]
        
        if e - s < 2:
            continue
            
        # Get the original n values (1-based)
        # sorted_indices contains 0-based indices into vals (which is sigma[1:])
        # So n = sorted_indices[idx] + 1
        group_idx = sorted_indices[s:e] + 1
        
        # Sort the n values to allow binary search
        group_idx.sort()
        
        L = len(group_idx)
        
        # Vectorized search for closest a to 1.5*b
        # targets = 1.5 * group_idx
        pos = np.searchsorted(group_idx, 1.5 * group_idx)
        
        # Candidates
        idx1 = np.clip(pos, 0, L - 1)
        idx2 = np.clip(pos - 1, 0, L - 1)
        
        a1 = group_idx[idx1]
        a2 = group_idx[idx2]
        b_arr = group_idx
        
        # Compute numerators |2a - 3b|
        num1 = np.abs(2 * a1 - 3 * b_arr)
        num2 = np.abs(2 * a2 - 3 * b_arr)
        
        # Masks for validity: a != b and 2a != 3b (num != 0)
        mask1 = (a1 != b_arr) & (num1 != 0)
        mask2 = (a2 != b_arr) & (num2 != 0)
        
        # We want to minimize num / b
        # Let's compute the float value for comparison
        # If mask is false, set to inf
        score1 = np.where(mask1, num1 / b_arr, np.inf)
        score2 = np.where(mask2, num2 / b_arr, np.inf)
        
        # Find the minimum in this group
        min_score1 = np.min(score1)
        min_score2 = np.min(score2)
        
        # Determine which candidate is better
        if min_score1 < min_score2:
            idx_min = np.argmin(score1)
            best_a_candidate = a1[idx_min]
            best_b_candidate = b_arr[idx_min]
            best_num_candidate = num1[idx_min]
            best_den_candidate = b_arr[idx_min]
            best_score_candidate = min_score1
        else:
            idx_min = np.argmin(score2)
            best_a_candidate = a2[idx_min]
            best_b_candidate = b_arr[idx_min]
            best_num_candidate = num2[idx_min]
            best_den_candidate = b_arr[idx_min]
            best_score_candidate = min_score2
            
        # Update global best if this is better
        # Check if valid (not inf)
        if best_score_candidate < float('inf'):
            # Compare exactly: best_num_candidate / best_den_candidate < best_num / best_den
            # best_num_candidate * best_den < best_num * best_den_candidate
            if best_num_candidate * best_den < best_num * best_den_candidate:
                best_error = best_score_candidate
                best_a = int(best_a_candidate)
                best_b = int(best_b_candidate)
                best_num = int(best_num_candidate)
                best_den = int(best_den_candidate)

    return [int(best_a), int(best_b)]
