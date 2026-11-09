import time
import numpy as np
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 10_000_000
    
    # Compute sigma(n) for all n up to N using sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
        
    if time.time() > deadline - 15:
        return [11, 6]

    # Optimize grouping using NumPy argsort
    # This is O(N log N) in C, much faster than Python loop
    sorted_indices = np.argsort(sigma)
    sorted_sigmas = sigma[sorted_indices]
    
    # Find boundaries where sigma value changes
    # unique_sigmas, counts = np.unique(sorted_sigmas, return_counts=True)
    # But we need the indices in sorted_indices corresponding to each unique sigma
    
    # Change points in sorted_sigmas
    # diff = np.diff(sorted_sigmas)
    # change_points = np.where(diff != 0)[0] + 1
    # starts = np.concatenate(([0], change_points))
    # ends = np.concatenate((change_points, [N]))
    
    # This might use a lot of memory for diff? N is 10^7. int64 diff is 80MB.
    # Let's try to avoid diff if possible, or just use it.
    # Actually, we can just iterate and check.
    
    # Alternative:
    # We can find unique values and their ranges.
    # But np.unique on 10^7 elements is fast.
    
    unique_sigmas, inverse, counts = np.unique(sigma, return_inverse=True, return_counts=True)
    # inverse[i] is the index in unique_sigmas for sigma[i]
    # This doesn't give us the sorted order of indices.
    
    # Let's stick to argsort.
    # sorted_indices: indices of original array that sort sigma.
    # sorted_sigmas: the sorted sigma values.
    
    # Find where sorted_sigmas changes
    # We can do this in a loop or vectorized.
    # Vectorized:
    diff = np.diff(sorted_sigmas)
    change_points = np.where(diff != 0)[0] + 1
    starts = np.concatenate(([0], change_points))
    ends = np.concatenate((change_points, [N]))
    
    if time.time() > deadline - 5:
        return [11, 6]

    best_err = float('inf')
    best_pair = None
    
    # Iterate through groups
    # We want to process groups with larger numbers first?
    # Pruning depends on max number in group.
    # So we should sort groups by max number descending.
    
    # Extract group maxes
    # group_maxes = np.array([sorted_indices[s:e][-1] for s, e in zip(starts, ends)]) # This is slow Python loop
    
    # Instead, just iterate. The number of unique sigmas is much smaller than N.
    # Probably < 10^6.
    
    # We can create a list of (start, end, max_val)
    group_ranges = []
    for s, e in zip(starts, ends):
        if e - s >= 2:
            # Get max value in this slice
            # sorted_indices[s:e] are the numbers
            max_val = np.max(sorted_indices[s:e])
            group_ranges.append((s, e, int(max_val)))
            
    # Sort by max_val descending
    group_ranges.sort(key=lambda x: x[2], reverse=True)
    
    for s, e, max_val in group_ranges:
        if time.time() > deadline - 2:
            break
            
        # Pruning
        if best_err < 1.0 / (2.0 * max_val):
            continue
            
        # Extract and sort the numbers in this group
        # They are not sorted by value in sorted_indices
        nums = sorted_indices[s:e].copy()
        nums.sort()
        
        n_len = len(nums)
        
        # Search for best pair
        # We can optimize this further?
        # For each b, find a closest to 1.5b.
        
        for b in nums:
            target = 1.5 * b
            idx = bisect.bisect_left(nums, target)
            
            if idx < n_len:
                a = nums[idx]
                if a != b:
                    num = abs(2*a - 3*b)
                    if num != 0:
                        err = num / (2*b)
                        if err < best_err:
                            best_err = err
                            best_pair = (a, b)
            if idx > 0:
                a = nums[idx-1]
                if a != b:
                    num = abs(2*a - 3*b)
                    if num != 0:
                        err = num / (2*b)
                        if err < best_err:
                            best_err = err
                            best_pair = (a, b)
                            
    if best_pair is not None:
        return [int(best_pair[0]), int(best_pair[1])]
    else:
        return [11, 6]
