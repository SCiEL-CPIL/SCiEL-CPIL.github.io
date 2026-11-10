import numpy as np
import time
from collections import defaultdict

def construct():
    N = 10**7
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    # 1. Precompute sigma
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        if time.time() > deadline - 30:  # Leave time for search
            break
        sigma[i::i] += i
    
    # 2. Build fiber dictionary: sigma value -> sorted list of numbers
    # Use a dictionary mapping sigma value to list of n
    fiber_dict = defaultdict(list)
    for n in range(1, N + 1):
        if time.time() > deadline - 20:
            break
        fiber_dict[sigma[n]].append(n)
    
    # Convert to sorted arrays for binary search
    # But this could use a lot of memory. Let's be careful.
    # Instead, for each fiber, sort it and store as a list.
    
    # 3. Search for best pair
    best_a = 1
    best_b = 1
    best_err = float('inf')
    
    # For each fiber, find the best pair (a, b) minimizing |a/b - 1.5|
    for s, nums in fiber_dict.items():
        if time.time() > deadline:
            break
        if len(nums) < 2:
            continue
        
        nums_sorted = sorted(nums)
        m = len(nums_sorted)
        
        # For each b in nums_sorted (descending), find a closest to 1.5*b
        # Use binary search
        for b in reversed(nums_sorted):
            # Lower bound on error for this b: even if a is perfect, error >= 0
            # But we can skip if 1.5*b is out of range
            target = 1.5 * b
            
            # Binary search for closest a in nums_sorted to target
            # Use bisect
            import bisect
            idx = bisect.bisect_left(nums_sorted, target)
            
            # Check idx and idx-1
            candidates = []
            if idx < m:
                candidates.append(nums_sorted[idx])
            if idx > 0:
                candidates.append(nums_sorted[idx - 1])
            
            for a in candidates:
                if a == b:
                    continue
                # Check |2a - 3b|
                diff = abs(2 * a - 3 * b)
                if diff == 0:
                    continue  # Invalid, error is 0
                err = diff / (2.0 * b)
                if err < best_err:
                    best_err = err
                    best_a = a
                    best_b = b
    
    # Fallback
    if best_a == 1 and best_b == 1:
        best_a = 11
        best_b = 6
    
    return [int(best_a), int(best_b)]
