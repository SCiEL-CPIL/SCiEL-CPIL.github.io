import numpy as np
import time
import bisect

def construct():
    start_time = time.time()
    MAX_N = 10_000_000
    deadline = start_time + 50.0
    
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    
    # Sieve
    for i in range(1, MAX_N + 1):
        sigma[i::i] += i
        # Time check
        if i % 1000000 == 0:
            if time.time() > deadline - 10:
                # Fallback to seed if sieve is too slow
                return [11, 6]

    # Group
    groups = {}
    for i in range(1, MAX_N + 1):
        s = int(sigma[i])
        if s in groups:
            groups[s].append(i)
        else:
            groups[s] = [i]

    # Search
    best_err = float('inf')
    best_a, best_b = 11, 6
    
    for s, nums in groups.items():
        if len(nums) < 2:
            continue
        nums.sort()
        
        # Optional: Prune if group is too small or spread out?
        # Not necessary.
        
        for b in nums:
            target = 1.5 * b
            idx = bisect.bisect_left(nums, target)
            
            candidates = []
            if idx < len(nums):
                candidates.append(nums[idx])
            if idx > 0:
                candidates.append(nums[idx-1])
                
            for a in candidates:
                if a == b:
                    continue
                if 2*a == 3*b:
                    continue
                err = abs(a / b - 1.5)
                if err < best_err:
                    best_err = err
                    best_a, best_b = a, b
                    # If we find a very small error, we could break early?
                    # But we need the global min.
                    # However, if err is 0 (handled) or very small, it's hard to beat.
                    # But we must check all to be sure.

    return [int(best_a), int(best_b)]
