import time
import numpy as np
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = 30
    L = 1
    for i in range(1, n + 1):
        L = L * i // gcd(L, i)
        
    weights = [L // (i + 1) for i in range(n)]
    
    half = 15
    left_weights = weights[:half]
    right_weights = weights[half:]
    
    # Generate left sums
    left_sums = [0]
    for w in left_weights:
        new_sums = []
        for s in left_sums:
            new_sums.append(s)
            new_sums.append(s + w)
            new_sums.append(s - w)
        left_sums = new_sums
        
    if time.time() > deadline:
         # Fallback if generation is too slow (unlikely)
         return [0]*28 + [1, -1]

    left_arr = np.array(left_sums, dtype=np.int64)
    left_arr.sort()
    
    # Generate right sums
    right_sums = [0]
    for w in right_weights:
        new_sums = []
        for s in right_sums:
            new_sums.append(s)
            new_sums.append(s + w)
            new_sums.append(s - w)
        right_sums = new_sums
            
    right_arr = np.array(right_sums, dtype=np.int64)
    
    # Vectorized search
    targets = -right_arr
    idx = np.searchsorted(left_arr, targets)
    
    # We need to check idx, idx-1, idx+1
    # Handle boundaries
    max_idx = len(left_arr)
    
    # Create arrays for candidates
    # We can just loop over the unique indices or vectorize the candidate selection
    # Vectorizing candidate selection is tricky due to bounds.
    # Let's create a list of candidate indices
    candidate_indices = []
    for i in range(len(right_arr)):
        p = idx[i]
        for delta in (-1, 0, 1):
            c = p + delta
            if 0 <= c < max_idx:
                candidate_indices.append(c)
                
    # This loop is 14M * 3 = 42M iterations. Might be slow in Python.
    # Alternative: Use numpy to handle bounds?
    # Or just accept the loop if it's fast enough.
    # 42M simple iterations in Python might take ~5-10s.
    # Total time budget is 80s. Generation ~10s. Sorting ~1s.
    # This should fit.
    
    best_abs = None
    best_left_val = 0
    best_right_val = 0
    best_left_idx = 0
    best_right_idx = 0
    
    # To speed up the candidate check, we can vectorize the value calculation
    # But we need the indices to reconstruct.
    
    # Let's optimize the candidate loop
    # We can process in chunks or just use a tight loop.
    
    # Actually, we can use numpy to get the values directly
    # But we need to handle the "closest" logic which depends on the specific target.
    
    # Let's stick to the loop but make it tight.
    
    # Optimization: 
    # We can compute the values for all idx, idx-1, idx+1 in bulk?
    # idx_clamped = np.clip(idx, 1, max_idx-1)
    # candidates = np.vstack([left_arr[idx_clamped-1], left_arr[idx_clamped], left_arr[idx_clamped+1]])
    # But we need to associate them with right_arr.
    
    # Let's try a semi-vectorized approach:
    # 1. Compute idx
    # 2. For each offset -1, 0, 1:
    #    - Calculate valid indices
    #    - Get left values
    #    - Compute sum with right values
    #    - Find min non-zero
    
    best_val = None
    best_l = 0
    best_r = 0
    
    for offset in [-1, 0, 1]:
        # Calculate candidate indices for this offset
        c_idx = idx + offset
        # Valid mask
        valid = (c_idx >= 0) & (c_idx < max_idx)
        if not np.any(valid):
            continue
            
        valid_indices = c_idx[valid]
        valid_right_vals = right_arr[valid]
        
        left_vals = left_arr[valid_indices]
        totals = left_vals + valid_right_vals
        
        # Mask out zeros
        nonzero_mask = totals != 0
        if not np.any(nonzero_mask):
            continue
            
        abs_totals = np.abs(totals[nonzero_mask])
        min_abs = np.min(abs_totals)
        
        if best_val is None or min_abs < best_val:
            best_val = min_abs
            # Find the index of the minimum in the masked array
            min_pos_in_masked = np.argmin(abs_totals)
            # Map back to original valid array index
            # We need to be careful with indices here.
            # Let's just find the index in the 'valid' set
            # Actually, np.argmin gives index in the filtered array.
            # We need the index in the 'valid' array to get the right value and left index.
            
            # To avoid complex index mapping, let's just store the values
            # We need left_val and right_val.
            # left_val = left_vals[nonzero_mask][min_pos_in_masked]
            # right_val = valid_right_vals[nonzero_mask][min_pos_in_masked]
            
            # But we also need to reconstruct coefficients.
            # We just need the target sums: best_left_sum and best_right_sum.
            
            # Get the specific values
            # The index in the original 'valid' array:
            # nonzero_mask is a boolean array of length len(valid)
            # min_pos_in_masked is the index in the compressed array
            # We can use np.flatnonzero to map back? Or just keep it simple.
            
            # Let's just extract the values
            filtered_left = left_vals[nonzero_mask]
            filtered_right = valid_right_vals[nonzero_mask]
            
            best_l = filtered_left[min_pos_in_masked]
            best_r = filtered_right[min_pos_in_masked]
            
    # Now we have best_l (integer sum) and best_r (integer sum)
    # Reconstruct coefficients
    
    def find_coeffs(weights, target):
        coeffs = [0] * len(weights)
        def dfs(i, current_sum):
            if i == len(weights):
                return current_sum == target
            w = weights[i]
            # Try 0
            coeffs[i] = 0
            if dfs(i+1, current_sum):
                return True
            # Try 1
            coeffs[i] = 1
            if dfs(i+1, current_sum + w):
                return True
            # Try -1
            coeffs[i] = -1
            if dfs(i+1, current_sum - w):
                return True
            coeffs[i] = 0
            return False
        if dfs(0, 0):
            return coeffs
        return None

    left_coeffs = find_coeffs(left_weights, best_l)
    right_coeffs = find_coeffs(right_weights, best_r)
    
    if left_coeffs is None or right_coeffs is None:
        return [0]*28 + [1, -1]
        
    return left_coeffs + right_coeffs
