import numpy as np
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s
    
    N = 10_000_000
    
    # Compute sigma(n) for all n up to N using numpy
    sigma = np.ones(N + 1, dtype=np.int64)
    
    for i in range(2, N + 1):
        if sigma[i] == 1:  # i is prime
            # For prime p, sigma(p^k) = (p^(k+1)-1)/(p-1)
            # We need to update multiples
            pass
    
    # This approach is complex. Let me use the divisor-sum sieve.
    # sigma[n] = sum of all divisors of n
    # For each d from 1 to N, add d to sigma[m] for all m that are multiples of d
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
    
    # Now group by sigma value
    # For efficiency, only consider numbers where a/b is close to 3/2
    # a ≈ 3b/2, so for each b, a should be near 3b/2
    
    # Create a dict: sigma_value -> list of n with that sigma
    # But this could use a lot of memory. Let's be smarter.
    
    # For each b from 1 to N, target a ≈ 3b/2
    # Check if there exists a near 3b/2 with sigma(a) = sigma(b)
    
    # Build lookup: for each sigma value, store the numbers
    # Use a dictionary
    sigma_to_nums = defaultdict(list)
    for n in range(1, N + 1):
        sigma_to_nums[int(sigma[n])].append(n)
    
    # Now search for best pair
    best_err = float('inf')
    best_pair = (1, 1)  # fallback
    
    # For b in range(1, N+1):
    #   target_a = 3*b/2
    #   Check a values near target_a in sigma_to_nums[sigma[b]]
    
    # To be efficient, for each sigma group, sort the numbers
    # Then for each b, binary search for a near 3b/2
    
    # But iterating all b is O(N) and for each we do a binary search - should be OK
    
    # Actually, let's iterate over b and check a few candidates near 3b/2
    # For each b, a must satisfy sigma(a) = sigma(b) and a is near 3b/2
    
    # Let's collect all numbers by sigma value, sorted
    for s_val, nums in sigma_to_nums.items():
        if len(nums) < 2:
            continue
        nums.sort()
        # For each b in nums, find a in nums closest to 3b/2
        for b in nums:
            target_a = 3 * b / 2.0
            # Binary search for target_a in nums
            import bisect
            idx = bisect.bisect_left(nums, target_a)
            candidates = []
            if idx < len(nums):
                candidates.append(nums[idx])
            if idx > 0:
                candidates.append(nums[idx - 1])
            for a in candidates:
                if a == b:
                    continue
                if a > N or b > N:
                    continue
                err = abs(a / b - 3.0 / 2.0)
                if err == 0:
                    continue
                if err < best_err:
                    best_err = err
                    best_pair = (a, b)
    
    # Check deadline
    if time.time() - start_time > DEADLINE:
        return [int(best_pair[0]), int(best_pair[1])]
    
    return [int(best_pair[0]), int(best_pair[1])]
