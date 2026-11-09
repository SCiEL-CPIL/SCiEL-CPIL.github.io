"""Approximate a ratio inside a divisor-sum fiber."""
import time
import math
from fractions import Fraction

# EVOLVE-BLOCK-START

def construct():
    """
    Find a, b <= 10^7 with sigma(a) = sigma(b) minimizing |a/b - 3/2|.
    """
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s limit
    
    MAX_N = 10_000_000
    TARGET_NUM, TARGET_DEN = 3, 2  # target ratio 3/2
    
    # We want a/b ≈ 3/2, so a ≈ 3b/2
    # sigma(a) = sigma(b)
    
    # Strategy:
    # 1. Compute sigma(n) for all n up to MAX_N using a sieve
    # 2. Group numbers by their sigma value
    # 3. For each group with at least 2 elements, find pairs (a,b) 
    #    that minimize |a/b - 3/2|
    
    # Step 1: Sieve for sigma
    sigma = [0] * (MAX_N + 1)
    for d in range(1, MAX_N + 1):
        for m in range(d, MAX_N + 1, d):
            sigma[m] += d
    
    # Check time
    if time.time() > deadline - 10:
        # Return a known valid pair
        return [int(11), int(6)]
    
    # Step 2: Group by sigma value
    from collections import defaultdict
    groups = defaultdict(list)
    for n in range(1, MAX_N + 1):
        s = sigma[n]
        groups[s].append(n)
    
    # Check time
    if time.time() > deadline - 5:
        return [int(11), int(6)]
    
    # Step 3: Find best pair
    best_a = 11
    best_b = 6
    best_err = abs(Fraction(11, 6) - Fraction(TARGET_NUM, TARGET_DEN))
    
    for s_val, nums in groups.items():
        if len(nums) < 2:
            continue
        
        # For each pair in this group, check |a/b - 3/2|
        # We want a/b ≈ 3/2, so a ≈ 1.5 * b
        # Sort nums to make it easier
        nums_sorted = sorted(nums)
        
        # For each b in nums_sorted, find a ≈ 1.5*b in the same group
        # Use binary search for efficiency
        for b in nums_sorted:
            target_a = 1.5 * b
            # Find closest elements to target_a in nums_sorted
            # Since we need a in the same group, we check neighbors
            
            # Binary search for target_a
            lo, hi = 0, len(nums_sorted) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if nums_sorted[mid] < target_a:
                    lo = mid + 1
                else:
                    hi = mid
            
            # Check lo and lo-1 (if valid)
            for idx in [lo - 1, lo]:
                if 0 <= idx < len(nums_sorted):
                    a = nums_sorted[idx]
                    if a == b:
                        continue
                    err = abs(Fraction(a, b) - Fraction(TARGET_NUM, TARGET_DEN))
                    if err < best_err:
                        best_err = err
                        best_a = a
                        best_b = b
                        if best_err == 0:
                            # Can't have exactly 0, but if we get very close
                            break
            
            # Early exit if time is running out
            if time.time() > deadline - 2:
                break
        
        if time.time() > deadline - 2:
            break
    
    # Return as Python ints in a list
    return [int(best_a), int(best_b)]

# EVOLVE-BLOCK-END
