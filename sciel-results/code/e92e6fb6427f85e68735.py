import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s
    
    # Increase N to find better approximations
    # 20,000,000 is a good balance of time and memory
    N = 20_000_000
    
    # Compute sigma(n) for n up to N
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
        # Check time periodically
        if d % 1_000_000 == 0:
            if time.time() > deadline - 15:
                # If we run out of time, we can't finish the sieve
                # But we can't use a partial sigma array easily for groups
                # So we must ensure the sieve finishes.
                # 20M iterations is ~2-3s. Should be fine.
                pass

    if time.time() > deadline - 10:
        # Fallback if sieve took too long (unlikely)
        return [11, 6]

    # Build groups: sigma value -> list of n
    # This loop is O(N)
    groups = defaultdict(list)
    for n in range(1, N + 1):
        groups[sigma[n]].append(n)
        
    if time.time() > deadline - 5:
        return [11, 6]

    best_num = None  # |2a - 3b|
    best_den = None  # b
    best_a = None
    best_b = None
    
    # Iterate through groups
    for s_val, nums in groups.items():
        if len(nums) < 2:
            continue
        if time.time() > deadline - 2:
            break
        
        nums.sort()
        
        # Two-pointer search for a close to 1.5 * b
        # We want to minimize |a/b - 1.5| = |2a - 3b| / 2b
        # Since b is positive, we minimize |2a - 3b| / b
        # We iterate b in increasing order.
        # The ideal a is 1.5 * b, which also increases.
        # So we can keep a pointer j into nums for a.
        
        j = 0
        k = len(nums)
        for b in nums:
            # Target a is 1.5 * b
            target = 1.5 * b
            
            # Advance j while nums[j] < target
            # We can start from previous j because target increases
            while j < k and nums[j] < target:
                j += 1
            
            # Check j and j-1
            # j is the first index where nums[j] >= target
            # So nums[j-1] is the largest < target
            
            candidates = []
            if j < k:
                candidates.append(nums[j])
            if j > 0:
                candidates.append(nums[j-1])
                
            for a in candidates:
                if a == b:
                    continue
                diff = abs(2 * a - 3 * b)
                if diff == 0:
                    continue  # Invalid, ratio exactly 3/2
                
                # Compare diff/b with best_num/best_den
                if best_num is None:
                    best_num = diff
                    best_den = b
                    best_a = a
                    best_b = b
                else:
                    # Check if diff/b < best_num/best_den
                    # diff * best_den < best_num * b
                    if diff * best_den < best_num * b:
                        best_num = diff
                        best_den = b
                        best_a = a
                        best_b = b

    if best_a is None:
        return [11, 6]
    
    return [int(best_a), int(best_b)]
