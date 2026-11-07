import time
import math
from fractions import Fraction
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    # Key insight: We need a/b ≈ 3/2, so a ≈ 1.5b.
    # Strategy:
    # 1. Precompute sigma for n up to a large limit using NumPy vectorized sieve.
    # 2. Group numbers by their sigma value.
    # 3. For each group with at least 2 elements, find the pair that minimizes |a/b - 3/2|.
    
    # Fallback pair: sigma(11) = 12, sigma(6) = 12
    best_a = 11
    best_b = 6
    best_error = Fraction(1, 3)
    
    # Use a large limit - NumPy makes this feasible
    L = 10_000_000
    
    try:
        # Compute sigma(n) for all n from 1 to L using a vectorized sieve approach
        sigma = np.zeros(L + 1, dtype=np.int64)
        
        # For each divisor d, add d to all multiples of d
        # Process in chunks to avoid creating huge temporary arrays
        chunk_size = 100000
        for d in range(1, L + 1):
            if time.time() - start_time > deadline - 10:
                break
            # Add d to sigma[d], sigma[2d], sigma[3d], ...
            multiples = np.arange(d, L + 1, d, dtype=np.int64)
            sigma[multiples] += d
        
        # Group by sigma value using a dictionary
        # Convert to Python list for easier handling
        sigma_list = sigma.tolist()
        
        groups = {}
        for n in range(1, L + 1):
            s = sigma_list[n]
            if s not in groups:
                groups[s] = []
            groups[s].append(n)
        
        # For each group with at least 2 elements, find best pair
        for s, nums in groups.items():
            if len(nums) < 2:
                continue
            
            if time.time() - start_time > deadline - 5:
                break
            
            nums_sorted = sorted(nums)
            
            # For each b in the group, find a ≈ 1.5*b
            for b in nums_sorted:
                target = 1.5 * b
                
                # Binary search for target in nums_sorted
                lo, hi = 0, len(nums_sorted)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if nums_sorted[mid] < target:
                        lo = mid + 1
                    else:
                        hi = mid
                
                # Check lo and lo-1 as candidates for a
                for idx in [lo, lo - 1]:
                    if 0 <= idx < len(nums_sorted):
                        a = nums_sorted[idx]
                        if a == b:
                            continue
                        # Compute |a/b - 3/2| exactly
                        err = abs(Fraction(a, b) - Fraction(3, 2))
                        if err < best_error:
                            best_error = err
                            best_a = a
                            best_b = b
        
    except (MemoryError, Exception):
        pass
    
    # Make sure we return Python ints
    return [int(best_a), int(best_b)]
