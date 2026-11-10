import numpy as np
import time
from collections import defaultdict

def construct():
    start = time.time()
    deadline = start + 80  # Safety margin before 90s limit

    N = 10_000_000

    # Phase 1: Compute sigma up to N using NumPy sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        if i % 500000 == 0:
            if time.time() > deadline - 45:
                N = i
                break

    if time.time() > deadline - 40:
        return [11, 6]

    # Phase 2: Group indices by sigma value
    sigma_vals = sigma[1:N+1]
    
    # Create a mapping from sigma value to list of indices
    # Use argsort to group efficiently
    sorted_indices = np.argsort(sigma_vals, kind='stable')
    sorted_sigmas = sigma_vals[sorted_indices]
    
    # Find boundaries of each group
    # Group boundaries are where sigma value changes
    diff = np.diff(sorted_sigmas)
    boundaries = np.concatenate([[0], np.where(diff != 0)[0] + 1, [len(sorted_indices)]])
    
    best_a, best_b = 11, 6
    best_num = abs(2*11 - 3*6)  # = 4
    best_den = 2*6  # = 12

    # Phase 3: For each fiber (group of numbers with same sigma), find best pair
    for g in range(len(boundaries) - 1):
        if time.time() > deadline - 15:
            break
        
        i_start = boundaries[g]
        i_end = boundaries[g+1]
        group_size = i_end - i_start
        
        if group_size < 2:
            continue
        
        # Get the actual numbers (indices + 1)
        nums = sorted_indices[i_start:i_end] + 1
        nums_sorted = np.sort(nums)
        nlen = len(nums_sorted)
        
        if nlen < 2:
            continue
        
        # Two-pointer technique to find pair minimizing |2a - 3b|
        # We want a/b close to 3/2, so 2a close to 3b
        # For each b, the best a is the one closest to 3b/2
        
        # Use binary search for each b to find closest a
        for k in range(nlen):
            b = int(nums_sorted[k])
            target_a = (3 * b) / 2.0
            
            # Binary search for closest to target_a
            lo, hi = 0, nlen - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if nums_sorted[mid] < target_a:
                    lo = mid + 1
                else:
                    hi = mid
            
            # Check lo and lo-1
            for idx in (lo, lo - 1):
                if 0 <= idx < nlen:
                    a = int(nums_sorted[idx])
                    if a == b:
                        continue
                    num = abs(2*a - 3*b)
                    den = 2*b
                    if num * best_den < best_num * den:
                        best_num = num
                        best_den = den
                        best_a, best_b = a, b

    # Phase 4: Targeted search using ratio constraint 2a - 3b = k
    # For small |k|, the error |a/b - 3/2| = |k|/(2b) is small when b is large.
    if time.time() < deadline - 5:
        for k in range(-50, 51):
            if time.time() > deadline - 5:
                break
            # a = (3b + k) / 2 must be integer
            # 3b + k even => b + k even
            if k % 2 != 0:
                # b must be odd
                start_b = 1
            else:
                # b must be even
                start_b = 2
            
            max_b = min(N, (2*N - k) // 3)
            min_b = max(start_b, (2 - k + 1) // 3)
            
            if min_b > max_b:
                continue
            
            # Adjust min_b for parity
            if k % 2 != 0:
                if min_b % 2 == 0:
                    min_b += 1
                step = 2
            else:
                if min_b % 2 != 0:
                    min_b += 1
                step = 2
            
            checks = 0
            for b in range(min_b, max_b + 1, step):
                if time.time() > deadline - 5:
                    break
                if checks > 200000:
                    break
                checks += 1
                a = (3 * b + k) // 2
                if a < 1 or a > N:
                    continue
                if a == b:
                    continue
                if sigma[a] == sigma[b]:
                    num = abs(2*a - 3*b)
                    den = 2*b
                    if num * best_den < best_num * den:
                        best_num = num
                        best_den = den
                        best_a, best_b = a, b

    return [int(best_a), int(best_b)]
