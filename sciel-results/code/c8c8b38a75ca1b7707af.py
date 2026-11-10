"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    deadline = time.time() + 80  # Safety margin before 90s limit
    
    N = 2000000  # Search up to this value for good pairs
    
    # Compute sigma(n) for all n up to N using sieve
    sigma = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigma[j] += i
    
    # Build dictionary: sigma_value -> list of n's
    from collections import defaultdict
    sigma_groups = defaultdict(list)
    for n in range(1, N + 1):
        sigma_groups[sigma[n]].append(n)
    
    # Find best pair minimizing |a/b - 3/2|
    target = Fraction(3, 2)
    best_error = None
    best_pair = (11, 6)  # Fallback: sigma(11)=12, sigma(6)=12, |11/6 - 3/2| = 1/3
    
    # Check groups with multiple elements
    for s_val, nums in sigma_groups.items():
        if len(nums) < 2:
            continue
        if time.time() > deadline - 5:
            break
        
        # Sort nums to enable efficient pair checking
        nums_sorted = sorted(nums)
        n_len = len(nums_sorted)
        
        # For each a, find b that minimizes |a/b - 3/2|
        # i.e., b ≈ 2a/3
        for i in range(n_len):
            a = nums_sorted[i]
            # We want b such that a/b ≈ 3/2, so b ≈ 2a/3
            target_b = 2 * a // 3
            
            # Binary search for target_b in nums_sorted
            lo, hi = 0, n_len - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if nums_sorted[mid] < target_b:
                    lo = mid + 1
                else:
                    hi = mid
            
            # Check lo and lo-1 (if valid)
            for j in [lo, lo - 1]:
                if 0 <= j < n_len:
                    b = nums_sorted[j]
                    if a == b:
                        continue
                    # Compute |a/b - 3/2| exactly
                    err = abs(Fraction(a, b) - target)
                    if err == 0:
                        continue
                    if best_error is None or err < best_error:
                        best_error = err
                        best_pair = (a, b)
    
    a, b = best_pair
    return [int(a), int(b)]

# EVOLVE-BLOCK-END
