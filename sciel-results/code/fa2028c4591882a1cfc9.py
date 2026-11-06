import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # 1. Calculate weights
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
        
    def lcm(a, b):
        return a * b // gcd(a, b)
        
    L = 1
    for i in range(1, 31):
        L = lcm(L, i)
        
    weights = np.array([L // i for i in range(1, 31)], dtype=np.int64)
    
    n = 30
    half = 15
    w1 = weights[:half]
    w2 = weights[half:]
    
    # 2. Generate sums for first half
    # arr: (N, 2) -> [sum, code]
    arr1 = np.array([[0, 0]], dtype=np.int64)
    for w in w1:
        # part -1
        p1 = arr1.copy()
        p1[:, 0] -= w
        p1[:, 1] = arr1[:, 1] * 3 + 0
        
        # part 0
        p2 = arr1.copy()
        # p2[:, 0] unchanged
        p2[:, 1] = arr1[:, 1] * 3 + 1
        
        # part 1
        p3 = arr1.copy()
        p3[:, 0] += w
        p3[:, 1] = arr1[:, 1] * 3 + 2
        
        arr1 = np.vstack([p1, p2, p3])
        
        if time.time() > deadline:
            break
            
    if time.time() > deadline:
        return [0] * 28 + [1, -1]

    # 3. Generate sums for second half
    arr2 = np.array([[0, 0]], dtype=np.int64)
    for w in w2:
        p1 = arr2.copy()
        p1[:, 0] -= w
        p1[:, 1] = arr2[:, 1] * 3 + 0
        
        p2 = arr2.copy()
        p2[:, 1] = arr2[:, 1] * 3 + 1
        
        p3 = arr2.copy()
        p3[:, 0] += w
        p3[:, 1] = arr2[:, 1] * 3 + 2
        
        arr2 = np.vstack([p1, p2, p3])
        
        if time.time() > deadline:
            break

    if time.time() > deadline:
        return [0] * 28 + [1, -1]

    # 4. Sort arr2 by sum
    arr2 = arr2[arr2[:, 0].argsort()]
    vals2 = arr2[:, 0]
    
    # 5. Search
    best_k = np.iinfo(np.int64).max
    best_code1 = 0
    best_code2 = 0
    
    # Iterate over arr1
    # To speed up, we can vectorize the search?
    # np.searchsorted can take an array of targets.
    targets = -arr1[:, 0]
    indices = np.searchsorted(vals2, targets)
    
    # We need to check indices and indices-1 for each row in arr1.
    # This is still a loop if we do it row by row, but we can vectorize the candidate selection.
    
    # Candidate indices: idx, idx-1
    # Clamp indices to valid range
    idxs = indices
    idxs_prev = indices - 1
    
    # Valid mask for idx
    mask1 = (idxs >= 0) & (idxs < len(vals2))
    # Valid mask for idx_prev
    mask2 = (idxs_prev >= 0) & (idxs_prev < len(vals2))
    
    # We will compute sums for valid candidates
    # We need to handle the case where sum is 0
    
    # Let's just loop, it might be fast enough with numpy inside?
    # No, 14 million iterations in Python is too slow.
    # We must vectorize the final check.
    
    # Create arrays for candidate sums
    # Candidate 1: idx
    valid_idx = np.where(mask1)[0]
    if len(valid_idx) > 0:
        cand_sums_1 = arr1[valid_idx, 0] + vals2[idxs[valid_idx]]
        cand_codes_1 = arr1[valid_idx, 1]
        cand_codes_2_1 = arr2[idxs[valid_idx], 1]
        
        # Exclude zero
        non_zero_1 = cand_sums_1 != 0
        valid_idx_1 = valid_idx[non_zero_1]
        cand_sums_1 = cand_sums_1[non_zero_1]
        
        if len(valid_idx_1) > 0:
            abs_sums_1 = np.abs(cand_sums_1)
            min_idx_1 = np.argmin(abs_sums_1)
            if abs_sums_1[min_idx_1] < best_k:
                best_k = abs_sums_1[min_idx_1]
                best_code1 = cand_codes_1[min_idx_1]
                best_code2 = cand_codes_2_1[min_idx_1]
                
    # Candidate 2: idx-1
    valid_idx_prev = np.where(mask2)[0]
    if len(valid_idx_prev) > 0:
        cand_sums_2 = arr1[valid_idx_prev, 0] + vals2[idxs_prev[valid_idx_prev]]
        cand_codes_2 = arr1[valid_idx_prev, 1]
        cand_codes_2_2 = arr2[idxs_prev[valid_idx_prev], 1]
        
        non_zero_2 = cand_sums_2 != 0
        valid_idx_2 = valid_idx_prev[non_zero_2]
        cand_sums_2 = cand_sums_2[non_zero_2]
        
        if len(valid_idx_2) > 0:
            abs_sums_2 = np.abs(cand_sums_2)
            min_idx_2 = np.argmin(abs_sums_2)
            if abs_sums_2[min_idx_2] < best_k:
                best_k = abs_sums_2[min_idx_2]
                best_code1 = cand_codes_2[min_idx_2]
                best_code2 = cand_codes_2_2[min_idx_2]

    # Fallback
    if best_k == np.iinfo(np.int64).max:
        return [0] * 28 + [1, -1]
        
    # 6. Reconstruct
    coeffs = [0] * 30
    
    temp_c = int(best_code1)
    for i in range(half - 1, -1, -1):
        digit = temp_c % 3
        temp_c //= 3
        coeffs[i] = digit - 1
        
    temp_c = int(best_code2)
    for i in range(half - 1, -1, -1):
        digit = temp_c % 3
        temp_c //= 3
        coeffs[half + i] = digit - 1
        
    return coeffs
