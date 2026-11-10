import numpy as np
import time
from collections import defaultdict
import bisect

def construct():
    start_time = time.time()
    DEADLINE = 55.0
    N = 10000000

    # 1. Vectorized sieve for sigma
    sigma = np.zeros(N + 1, dtype=np.int64)
    # sigma[1] will be 1 after d=1 iteration
    for d in range(1, N + 1):
        if d % 1000000 == 0 and time.time() - start_time > DEADLINE - 10:
            # Fallback if sieve is too slow
            return [11, 6]
        sigma[d::d] += d

    if time.time() - start_time > DEADLINE - 10:
        return [11, 6]

    # 2. Group numbers by sigma value
    groups = defaultdict(list)
    sigma_list = sigma.tolist()
    
    for i in range(1, N + 1):
        if time.time() - start_time > DEADLINE - 5:
            break
        groups[sigma_list[i]].append(i)

    # 3. Search for best pair
    best_error = abs(11/6 - 1.5)
    best_a = 11
    best_b = 6
    
    for s, nums in groups.items():
        if len(nums) < 2:
            continue
        
        if time.time() - start_time > DEADLINE - 2:
            break
            
        nums.sort()
        
        min_b_needed = int(1.0 / (2.0 * best_error)) + 1
        
        for b in nums:
            if b < min_b_needed:
                continue
            
            target_a = 1.5 * b
            idx = bisect.bisect_left(nums, target_a)
            
            candidates = []
            if idx < len(nums):
                candidates.append(nums[idx])
            if idx > 0:
                candidates.append(nums[idx-1])
                
            for a in candidates:
                if a == b:
                    continue
                
                diff = abs(2 * a - 3 * b)
                if diff == 0:
                    continue
                
                err = diff / (2.0 * b)
                if err < best_error:
                    best_error = err
                    best_a = a
                    best_b = b
                    min_b_needed = int(1.0 / (2.0 * best_error)) + 1

    return [int(best_a), int(best_b)]
