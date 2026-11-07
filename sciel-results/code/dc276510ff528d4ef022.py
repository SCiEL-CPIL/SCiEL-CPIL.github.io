import bisect
import time
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit to be safe
    
    N = 10**7
    
    # 1. Compute sigma for all n up to N
    # Using a standard sieve
    sigma = [0] * (N + 1)
    for i in range(1, N + 1):
        # Optimization: break if time is running out, but we need full sigma for correctness
        # If we don't finish, we can't do the group search correctly.
        # However, 10^7 is likely fast enough.
        for j in range(i, N + 1, i):
            sigma[j] += i
            
    # Check time
    if time.time() > deadline:
        # Fallback to a known good pair if sieve takes too long
        # (122, 80) is a known good pair with error 0.025
        return [122, 80]

    # 2. Group numbers by sigma value
    sigma_to_nums = defaultdict(list)
    for n in range(1, N + 1):
        sigma_to_nums[sigma[n]].append(n)
        
    if time.time() > deadline:
        return [122, 80]

    # 3. Search for best pair
    best_error = float('inf')
    best_pair = (122, 80) # Fallback
    
    # Iterate through groups
    for s, nums in sigma_to_nums.items():
        if len(nums) < 2:
            continue
        
        # Sort the numbers in this group to allow binary search
        nums.sort()
        
        # For each a in the group, find b such that a/b is close to 1.5
        # b should be close to a / 1.5 = 2a/3
        for a in nums:
            target_b = (2 * a) / 3.0
            
            # Find position in sorted nums
            pos = bisect.bisect_left(nums, target_b)
            
            # Check neighbors around pos
            # We check pos-1, pos, pos+1 to be safe
            candidates = []
            if pos > 0:
                candidates.append(nums[pos-1])
            if pos < len(nums):
                candidates.append(nums[pos])
            if pos + 1 < len(nums):
                candidates.append(nums[pos+1])
                
            for b in candidates:
                if b == 0:
                    continue
                # Calculate error
                # We want to minimize |a/b - 1.5|
                # Using integer arithmetic to avoid float precision issues in comparison?
                # |a/b - 3/2| = |2a - 3b| / (2b)
                # We want to minimize this fraction.
                # Since b is positive, we minimize |2a - 3b| / b
                # Let's just compute the float value, it's safe for 10^7
                ratio = a / b
                error = abs(ratio - 1.5)
                
                if error < best_error:
                    best_error = error
                    best_pair = (a, b)
                    
        # Optional: Early termination if error is very small?
        # The minimum possible non-zero error is 1/LCM(a,b) roughly.
        # For 10^7, LCM can be large. 1/10^7 is 1e-7.
        # If best_error < 1e-5, we might stop?
        # But we don't know if a better one exists.
        
    return [best_pair[0], best_pair[1]]
