"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from fractions import Fraction

MAX_N = 10_000_000
TIME_LIMIT = 85  # seconds, safety margin before 90s kill

def construct():
    start_time = time.time()
    
    # Step 1: Compute sigma(n) for all n up to MAX_N using sieve
    sigma = [0] * (MAX_N + 1)
    for i in range(1, MAX_N + 1):
        for j in range(i, MAX_N + 1, i):
            sigma[j] += i
    
    # Step 2: Group numbers by their sigma value
    # Use a dict: sigma_value -> list of n values
    groups = {}
    for n in range(1, MAX_N + 1):
        s = sigma[n]
        if s not in groups:
            groups[s] = []
        groups[s].append(n)
    
    # Step 3: For each group, find the pair (a, b) minimizing |2a - 3b|
    # Use two-pointer technique on the sorted list
    best_diff = None  # |2a - 3b| as integer
    best_a = None
    best_b = None
    
    for s, nums in groups.items():
        if len(nums) < 2:
            continue
        # nums is already sorted (we appended in order)
        # Two-pointer: for each a, find b closest to 2a/3
        # We want to minimize |2a - 3b|
        # For a given a, optimal b is near 2a/3
        # Use binary search or two-pointer
        
        # Two-pointer approach:
        # Sort nums (already sorted)
        # For each a in nums, find b in nums closest to 2*a/3
        # Since nums is sorted, we can use a pointer that moves monotonically
        
        n = len(nums)
        # For each a, the target b is 2a/3
        # As a increases, target b increases, so pointer moves forward
        ptr = 0
        for i in range(n):
            a = nums[i]
            target = 2 * a / 3.0
            # Move pointer to find b closest to target
            while ptr < n - 1 and nums[ptr + 1] <= target:
                ptr += 1
            # Check ptr and ptr+1 (if in range)
            for p in (ptr, min(ptr + 1, n - 1)):
                b = nums[p]
                diff = abs(2 * a - 3 * b)
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_a = a
                    best_b = b
        
        # Time check
        if time.time() - start_time > TIME_LIMIT - 5:
            break
    
    if best_a is None:
        return [1, 1]
    
    return [int(best_a), int(best_b)]

# EVOLVE-BLOCK-END
