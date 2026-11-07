import time
import math
import numpy as np
from collections import defaultdict
import bisect

def construct():
    start_time = time.time()
    deadline = 85.0  # Leave 5s margin
    
    N = 10_000_000
    
    # 1. Compute sigma(n) for all n <= N using additive sieve
    # sigma[n] = sum of divisors of n
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Vectorized additive sieve
    # This loop runs N times, but the inner operation is vectorized C code.
    # Total operations ~ N log N.
    for i in range(1, N + 1):
        sigma[i::i] += i
        if i % 100000 == 0:
            if time.time() - start_time > deadline - 40:
                # Fallback to hardcoded if sieve is too slow
                return [11, 6]
    
    if time.time() - start_time > deadline - 30:
        return [11, 6]
    
    # 2. Group numbers by their sigma value
    # Using a dictionary of lists
    groups = defaultdict(list)
    # Convert numpy array to list for faster iteration? Or iterate over numpy array?
    # Iterating over numpy array is slower than list in Python 3 sometimes, but let's try.
    # Actually, sigma is a numpy array.
    for n in range(1, N + 1):
        groups[int(sigma[n])].append(n)
        
    if time.time() - start_time > deadline - 15:
        return [11, 6]
    
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    # 3. Search for pairs
    # We want to minimize |a/b - 1.5| subject to sigma(a)=sigma(b) and a!=b (if a/b=1.5 exactly is invalid? 
    # "A raw value of exactly 0 cannot be scored and counts as invalid.")
    # So if 2a == 3b, we must skip that pair.
    
    # Sort groups by size descending to find good candidates faster?
    # Or just iterate all.
    # Large groups have more chances of having close ratios.
    sorted_groups = sorted(groups.items(), key=lambda x: -len(x[1]))
    
    for s_val, nums in sorted_groups:
        if time.time() - start_time > deadline:
            break
            
        if len(nums) < 2:
            continue
            
        nums.sort()
        
        # For each b in nums, find a in nums closest to 1.5 * b
        # We can optimize by noticing that if we iterate b, a is determined.
        # But doing binary search for every b is O(K log K).
        # For K up to a few thousand, this is fine.
        
        for b in nums:
            if time.time() - start_time > deadline:
                break
                
            target = 1.5 * b
            # Find position where target would be inserted
            idx = bisect.bisect_left(nums, target)
            
            # Check idx and idx-1
            for j in [idx, idx - 1]:
                if 0 <= j < len(nums):
                    a = nums[j]
                    if a == b:
                        continue
                    # Check if ratio is exactly 1.5
                    if 2 * a == 3 * b:
                        continue
                    
                    err = abs(2 * a - 3 * b) / (2.0 * b)
                    if err < best_error:
                        best_error = err
                        best_a = a
                        best_b = b
                        
    if best_error == float('inf'):
        return [11, 6]
        
    return [int(best_a), int(best_b)]
