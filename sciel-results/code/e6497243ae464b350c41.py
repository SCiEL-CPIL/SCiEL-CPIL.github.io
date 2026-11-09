import numpy as np
import bisect
import time

def construct():
    N = 10000000
    start_time = time.time()
    deadline = start_time + 55  # 5s margin
    
    # 1. Compute sigma(n) for n in [1, N]
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve for sigma
    # This loop might be slow in Python, but NumPy handles the vector part.
    for i in range(1, N + 1):
        if time.time() > deadline:
            # Fallback to seed if too slow
            return [11, 6]
        sigma[i::i] += i
        
    # 2. Group by sigma value
    groups = {}
    for n in range(1, N + 1):
        s = int(sigma[n])
        if s not in groups:
            groups[s] = []
        groups[s].append(n)
        
    # 3. Search for best pair
    best_err = float('inf')
    best_a = 11
    best_b = 6
    
    # Pre-sort lists? Or sort on the fly?
    # Sorting on the fly is better as we only sort groups we visit.
    # But we visit all groups.
    # Total cost of sorting is sum(|G| log |G|).
    
    for s, nums in groups.items():
        if time.time() > deadline:
            break
        if len(nums) < 2:
            continue
            
        nums.sort()
        
        # For each a in nums, find b closest to 2a/3
        # We can optimize by noticing that if a is very large, b must be large.
        # But we just bisect.
        
        for a in nums:
            target_b = (2 * a) / 3.0
            
            # Find insertion point
            idx = bisect.bisect_left(nums, target_b)
            
            # Check neighbors
            # idx can be 0 or len(nums)
            candidates = []
            if idx < len(nums):
                candidates.append(nums[idx])
            if idx > 0:
                candidates.append(nums[idx - 1])
                
            for b in candidates:
                if b == 0:
                    continue
                # Calculate error
                # a/b - 1.5
                err = abs(a / b - 1.5)
                if err < best_err:
                    best_err = err
                    best_a = a
                    best_b = b
                    
    return [int(best_a), int(best_b)]
