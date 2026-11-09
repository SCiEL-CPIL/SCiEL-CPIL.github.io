import numpy as np
import time

def construct():
    t_start = time.time()
    deadline = t_start + 85.0  # Safety margin
    
    MAX_N = 10_000_000
    
    # 1. Compute sigma(n) for n=1..MAX_N using NumPy sieve
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    
    # The loop d=1 to MAX_N
    # We can unroll or use a faster method?
    # This is the standard fast way in NumPy.
    for d in range(1, MAX_N + 1):
        if time.time() > deadline - 20:
            # Fallback to a known valid pair
            # sigma(6) = 1+2+3+6 = 12
            # sigma(11) = 1+11 = 12
            # a=11, b=6 -> |11/6 - 1.5| = |1.833 - 1.5| = 0.333
            # a=6, b=11 -> |6/11 - 1.5| = |0.545 - 1.5| = 0.95
            # So [11, 6] is better.
            return [11, 6]
        sigma[d:MAX_N+1:d] += d
        
    if time.time() > deadline - 10:
        return [11, 6]

    # 2. Group by sigma value
    # We only care about indices 1..MAX_N
    sigma_valid = sigma[1:]
    
    # Get indices that sort sigma_valid
    # np.argsort returns indices that would sort the array
    sorted_indices = np.argsort(sigma_valid)
    sorted_sigma = sigma_valid[sorted_indices]
    
    if time.time() > deadline - 5:
        return [11, 6]

    # Find boundaries of groups with identical sigma values
    # diff != 0 indicates a change in sigma value
    diff = np.diff(sorted_sigma)
    change_points = np.where(diff != 0)[0] + 1
    starts = np.concatenate(([0], change_points))
    ends = np.concatenate((change_points, [len(sigma_valid)]))
    
    best_err = float('inf')
    best_a = 11
    best_b = 6
    
    # 3. Iterate over groups
    # Number of groups is len(starts)
    for i in range(len(starts)):
        if time.time() > deadline - 2:
            break
            
        s_start = starts[i]
        s_end = ends[i]
        group_size = s_end - s_start
        
        if group_size < 2:
            continue
            
        # Get the actual numbers (indices in 1..MAX_N)
        # sorted_indices contains indices 0..MAX_N-1 corresponding to 1..MAX_N
        # So we add 1
        nums = sorted_indices[s_start:s_end] + 1
        # Sort them to allow binary search
        nums_sorted = np.sort(nums)
        
        # Pruning: If the smallest possible error for the largest b in this group
        # is >= best_err, skip the group.
        # Min error for b is approx 1/(2b).
        # For b = nums_sorted[-1], min_err = 1/(2 * nums_sorted[-1])
        if 1.0 / (2.0 * nums_sorted[-1]) >= best_err:
            continue
            
        # Search for best pair in this group
        # Iterate b in descending order to find large b (small error) first
        # This allows early termination if min_err_b >= best_err
        for b in reversed(nums_sorted):
            # Calculate minimum possible error for this b
            # If b is odd, min |2a-3b| is 1 (since 3b odd, 2a even, diff odd). Err = 1/(2b).
            # If b is even, min |2a-3b| is 2 (since 3b even, 2a even, diff even, non-zero). Err = 2/(2b) = 1/b.
            if b % 2 != 0:
                min_err_b = 1.0 / (2.0 * b)
            else:
                min_err_b = 1.0 / b
                
            if min_err_b >= best_err:
                # Since we are iterating b in descending order,
                # subsequent b's are smaller, so min_err_b will be larger.
                # We can break.
                break
                
            target_a = 1.5 * b
            # Find position of target_a in nums_sorted
            pos = np.searchsorted(nums_sorted, target_a)
            
            # Check candidates around pos
            # We check a small window
            # pos is the index where target_a would be inserted
            # Candidates are at pos-1, pos, pos+1, etc.
            # We need to ensure indices are valid.
            
            # To avoid duplicates and ensure we check the closest ones,
            # we can check a range of indices.
            start_idx = max(0, pos - 5)
            end_idx = min(len(nums_sorted), pos + 6)
            
            for idx_a in range(start_idx, end_idx):
                a = nums_sorted[idx_a]
                
                # Calculate error
                num = abs(2 * a - 3 * b)
                if num == 0:
                    continue # Invalid: error exactly 0
                
                den = 2 * b
                err = num / den
                
                if err < best_err:
                    best_err = err
                    best_a = a
                    best_b = b
                    
    if best_err == float('inf'):
        return [11, 6]
        
    return [int(best_a), int(best_b)]
