import time
import numpy as np
from collections import defaultdict
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin within 90s wall-clock
    
    N = 10_000_000
    
    # 1. Compute sigma(n) for n in [1, N] using a vectorized sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    for d in range(1, N + 1):
        sigma[d::d] += d
        
        if d % 2_000_000 == 0:
            if time.time() > deadline - 15:
                break
    
    # 2. Group numbers by sigma value
    # Only keep groups with at least 2 elements to save memory/time
    groups = defaultdict(list)
    
    for n in range(1, N + 1):
        s = sigma[n]
        if s > 0:
            groups[s].append(n)
    
    if time.time() > deadline - 10:
        return [11, 6]
    
    # 3. Search for best pair minimizing |a/b - 3/2|
    # Error = |2a - 3b| / (2b), must be non-zero (a != 3b/2 in rational sense, i.e., 2a != 3b)
    best_a = 11
    best_b = 6
    best_num = abs(2 * 11 - 3 * 6)  # 4
    best_den = 2 * 6  # 12
    
    # Sort groups by size descending so we process larger groups first (more chances to find good pairs)
    # But actually, we want to find the global minimum. Let's just iterate all groups.
    # To be efficient, for each group we do a binary search for each element.
    
    # Process groups in order of size (larger groups first) to find good candidates early
    # This allows early termination if we find a very good pair
    sorted_groups = sorted(groups.items(), key=lambda x: -len(x[1]))
    
    for s, nums in sorted_groups:
        if len(nums) < 2:
            continue
        
        # nums is already sorted (built by iterating n from 1 to N)
        # For each a in nums, find b in nums such that b ≈ 2a/3
        # We want to minimize |2a - 3b| / (2b)
        
        for a in nums:
            target = (2.0 * a) / 3.0
            
            idx = bisect_left(nums, target)
            
            candidates = []
            if idx < len(nums):
                candidates.append(nums[idx])
            if idx > 0:
                candidates.append(nums[idx - 1])
            
            for b in candidates:
                if b == a:
                    continue
                
                diff = abs(2 * a - 3 * b)
                if diff == 0:
                    continue  # Error is exactly 0, invalid
                
                den = 2 * b
                
                # Compare diff/den < best_num/best_den
                if diff * best_den < best_num * den:
                    best_num = diff
                    best_den = den
                    best_a = a
                    best_b = b
                    
        # Check time
        if time.time() > deadline - 5:
            break
    
    return [int(best_a), int(best_b)]
