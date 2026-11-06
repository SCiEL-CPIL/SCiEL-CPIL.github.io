import time
import numpy as np
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = 30
    mid = 15
    
    # Calculate LCM of 1..30
    def lcm(a, b):
        return a * b // gcd(a, b)
    L = reduce(lcm, range(1, n + 1))
    
    # Weights: L / i for i in 1..30
    # i goes from 1 to 30
    weights = np.array([L // i for i in range(1, n + 1)], dtype=np.int64)
    
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    # Generate all sums for a set of weights
    def generate_sums(weights_half):
        # Number of combinations
        num_combos = 3 ** len(weights_half)
        # We need to generate the base-3 digits for 0 to num_combos - 1
        # This is a standard "unary to base-3" conversion vectorized.
        
        # Create an array of indices 0 to num_combos - 1
        indices = np.arange(num_combos, dtype=np.uint64)
        
        # We will build the digits matrix
        # D has shape (num_combos, len(weights_half))
        # D[:, j] is the j-th digit (0-indexed from right)
        
        # To avoid huge temporary arrays, we can compute sums iteratively?
        # No, vectorized dot product is faster.
        # Let's create the digits matrix.
        
        # Power of 3 for each position
        powers = 3 ** np.arange(len(weights_half))
        
        # D[:, j] = (indices // powers[j]) % 3
        # This creates len(weights_half) arrays of size num_combos.
        # 15 * 14M * 8 bytes = 1.6 GB. Too much memory!
        
        # Alternative: Compute sums iteratively.
        sums = np.array([0], dtype=np.int64)
        for w in weights_half:
            # sums_new = [s-w, s, s+w for s in sums]
            # This triples the size.
            # Memory peak: 3 * current_size.
            # Final size 14M.
            # Peak during last step: 3 * (14M/3) = 14M.
            # So peak is roughly 2 * final size.
            # 2 * 14M * 8 bytes = 224 MB. Acceptable.
            sums = np.concatenate([sums - w, sums, sums + w])
        return sums

    # Generate sums for both halves
    sums1 = generate_sums(w1)
    sums2 = generate_sums(w2)
    
    # Sort sums2 for search
    sums2_sorted = np.sort(sums2)
    
    # Find best pair
    # We want min |s1 + s2| != 0
    # For each s1, target is -s1.
    
    # Vectorized search
    targets = -sums1
    # searchsorted returns index where target would be inserted
    # We check idx and idx-1
    indices = np.searchsorted(sums2_sorted, targets)
    
    len_s2 = len(sums2_sorted)
    
    # Candidate 1: index at indices
    idx_r = np.clip(indices, 0, len_s2 - 1)
    s2_r = sums2_sorted[idx_r]
    total_r = sums1 + s2_r
    
    # Candidate 2: index at indices - 1
    idx_l = np.clip(indices - 1, 0, len_s2 - 1)
    s2_l = sums2_sorted[idx_l]
    total_l = sums1 + s2_l
    
    # We need min absolute non-zero total
    # If total is 0, we ignore it.
    
    # Combine candidates
    # We can just take the min of abs(total_r) and abs(total_l) where total != 0
    
    # To handle the "zero trap" (if both are 0, we missed the next one),
    # note that if s1+s2=0, then -s1 is exactly in sums2.
    # searchsorted will point to the first occurrence.
    # If there are duplicates of -s1, idx_r points to the first, idx_l points to the previous (which might be -s1 too if duplicates, or different).
    # If the value at idx_r is 0-sum, and idx_l is also 0-sum, we need idx_r+1 or idx_l-1.
    
    # Let's identify the global minimum non-zero absolute value.
    
    # Mask zeros
    mask_r = total_r != 0
    mask_l = total_l != 0
    
    abs_r = np.abs(total_r)
    abs_l = np.abs(total_l)
    
    # Set zeros to infinity
    abs_r[~mask_r] = np.iinfo(np.int64).max
    abs_l[~mask_l] = np.iinfo(np.int64).max
    
    # Find global min
    min_val_r = np.min(abs_r)
    min_val_l = np.min(abs_l)
    
    if min_val_r < min_val_l:
        best_abs_S = min_val_r
        idx_best_in_sums1 = np.argmin(abs_r)
        best_s1 = sums1[idx_best_in_sums1]
        best_s2 = s2_r[idx_best_in_sums1]
    else:
        best_abs_S = min_val_l
        idx_best_in_sums1 = np.argmin(abs_l)
        best_s1 = sums1[idx_best_in_sums1]
        best_s2 = s2_l[idx_best_in_sums1]
        
    # Check if we hit the zero trap: if best_abs_S is inf, it means all candidates were 0?
    # This implies s1+s2=0 for all checked pairs? Unlikely.
    # But what if the true min was at idx_r+1?
    # If total_r == 0 and total_l == 0, then abs_r and abs_l were set to INF.
    # Then min_val would be INF.
    # In that case, we need to look at idx_r+1 or idx_l-1.
    
    if best_abs_S == np.iinfo(np.int64).max:
        # Fallback: check neighbors of the zero-sums
        # This case is rare but possible.
        # We know that for some s1, the closest s2 sums to 0.
        # Let's find an s1 that had total_r == 0.
        zero_mask_r = (total_r == 0)
        if np.any(zero_mask_r):
            # Pick one such s1
            idx_zero = np.where(zero_mask_r)[0][0]
            s1_val = sums1[idx_zero]
            target = -s1_val
            # Find range of target in sums2_sorted
            left_idx = np.searchsorted(sums2_sorted, target, side='left')
            right_idx = np.searchsorted(sums2_sorted, target, side='right')
            
            # Check neighbors
            candidates = []
            if left_idx > 0:
                candidates.append(sums2_sorted[left_idx - 1])
            if right_idx < len_s2:
                candidates.append(sums2_sorted[right_idx])
                
            best_local_abs = np.iinfo(np.int64).max
            best_s2_val = 0
            for s2_cand in candidates:
                t = s1_val + s2_cand
                if t != 0:
                    if abs(t) < best_local_abs:
                        best_local_abs = abs(t)
                        best_s2_val = s2_cand
            
            if best_local_abs < best_abs_S:
                best_abs_S = best_local_abs
                best_s1 = s1_val
                best_s2 = best_s2_val
                
    # Now we have best_s1 and best_s2.
    # We need to reconstruct coefficients.
    
    # Find index in sums1 that corresponds to best_s1
    # sums1 is unsorted.
    idx1 = np.where(sums1 == best_s1)[0][0]
    
    # Find index in sums2 (original, unsorted) that corresponds to best_s2
    # Note: sums2_sorted is sorted, but we need the index in the original generation order to decode?
    # No, the index in the *generation* order (0 to 3^15-1) corresponds to the base-3 digits.
    # The `generate_sums` function generates sums in a specific order.
    # Does the index in `sums1` correspond to the base-3 index?
    # Let's trace `generate_sums`:
    # Start: [0]
    # Step 1: [-w0, 0, w0] -> Indices 0, 1, 2 correspond to coeffs -1, 0, 1 for w0.
    # Step 2: For each s in prev:
    #   s-w1, s, s+w1
    #   This appends -1, 0, 1 for w1 to the coeffs of the parent.
    #   So the index is indeed the base-3 number where the digit for w0 is the least significant?
    #   Let's check:
    #   Index 0: -1 for w0, then -1 for w1... -> All -1.
    #   Index 1: -1 for w0, then 0 for w1...
    #   Index 2: -1 for w0, then 1 for w1...
    #   Index 3: 0 for w0, then -1 for w1...
    #   Yes, the index is the base-3 representation where digit 0=-1, 1=0, 2=1.
    #   Wait, standard base 3 uses 0,1,2.
    #   My mapping: 0->-1, 1->0, 2->1.
    #   So index I:
    #   I // 1 % 3 -> coeff for w0
    #   I // 3 % 3 -> coeff for w1
    #   etc.
    
    # So we just need to decode idx1 and idx2.
    
    def decode_index(idx, num_weights):
        coeffs = []
        for i in range(num_weights):
            digit = (idx // (3 ** i)) % 3
            # Map 0->-1, 1->0, 2->1
            if digit == 0:
                coeffs.append(-1)
            elif digit == 1:
                coeffs.append(0)
            else:
                coeffs.append(1)
        return coeffs
    
    c1 = decode_index(idx1, mid)
    c2 = decode_index(idx2, mid)
    
    final_coeffs = c1 + c2
    
    # Verify
    # val = sum(c[i] / (i+1) for i in range(30))
    # assert val != 0
    
    return final_coeffs
