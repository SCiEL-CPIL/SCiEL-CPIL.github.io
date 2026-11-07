import numpy as np
import bisect
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin out of 60s
    
    N = 10000000
    # Compute sigma(n) for all n up to N
    # sigma(n) = sum of divisors of n
    # Using a sieve approach
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        # Add i to sigma[j] for all multiples j of i
        # sigma[i::i] += i
        # This is efficient in NumPy
        sigma[i::i] += i
        
    # Check time
    if time.time() > deadline:
        # Fallback to a known good pair if sieve takes too long (unlikely for 10^7)
        # But let's try to return something valid immediately if we are close to deadline
        # Actually, sieve for 10^7 is fast (~1-2s).
        pass

    # Group indices by sigma value
    # We need a dictionary: sigma_val -> list of n
    # Using Python dictionary and lists
    sigma_groups = {}
    for n in range(1, N + 1):
        s = int(sigma[n])
        if s not in sigma_groups:
            sigma_groups[s] = []
        sigma_groups[s].append(n)
        
    # Check time
    if time.time() > deadline:
        return [11, 6] # Fallback

    best_a, best_b = 11, 6
    best_err = abs(11/6 - 1.5)
    
    # For each group, find the pair minimizing |a/b - 1.5|
    for s, nums in sigma_groups.items():
        if len(nums) < 2:
            continue
        
        # nums is already sorted because we iterated n in increasing order
        # For each b in nums, find a in nums closest to 1.5 * b
        for b in nums:
            target = 1.5 * b
            # Binary search for target in nums
            # We want a such that a/b is close to 1.5 => a is close to 1.5*b
            # bisect_left returns index where target would be inserted
            idx = bisect.bisect_left(nums, int(target))
            
            # Check neighbors: idx-1 and idx
            candidates = []
            if idx > 0:
                candidates.append(nums[idx-1])
            if idx < len(nums):
                candidates.append(nums[idx])
                
            for a in candidates:
                if a == b:
                    continue
                err = abs(a/b - 1.5)
                if err < best_err:
                    best_err = err
                    best_a, best_b = a, b
                    
        # Check time
        if time.time() > deadline:
            break

    return [int(best_a), int(best_b)]
