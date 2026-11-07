import time
import sys
from fractions import Fraction
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    """
    Find [a, b] positive ints <= 1e7 such that sigma(a) = sigma(b),
    minimizing |a/b - 3/2|.
    
    Strategy:
    1. Compute sigma(n) for all n up to N=10^7 using a sieve.
    2. Group numbers by their sigma value.
    3. For each group with >= 2 elements, find the pair (a, b) that minimizes |a/b - 3/2|.
       Since we want a/b ≈ 3/2, i.e., a ≈ 1.5*b, for each b in the group,
       we want a close to 1.5*b. We can binary search in the sorted list of the group.
    4. Track the global best pair.
    
    To fit within time limits:
    - Sieve sigma up to 10^7.
    - Group by sigma.
    - For each group, sort the list (or keep sorted during construction).
    - For each element b, binary search for target = 1.5 * b.
    - Compute exact error using Fraction or integer arithmetic.
    """
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    N = 10_000_000
    
    # Step 1: Sieve sigma(n) for n in [1, N]
    # sigma(n) = sum of divisors. Use the standard sieve approach.
    sigma = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigma[j] += i
    
    # Step 2: Group by sigma value
    groups = defaultdict(list)
    for n in range(1, N + 1):
        s = sigma[n]
        groups[s].append(n)
    
    # Step 3: For each group, find best pair
    best_error = Fraction(1, 1)  # large initial error
    best_pair = (1, 1)
    
    # Sort each group once
    for s, lst in groups.items():
        if len(lst) < 2:
            continue
        lst.sort()
        
        # For each b in lst, we want a such that a/b ≈ 3/2, i.e., a ≈ 3b/2
        # We'll iterate through b and binary search for a near 3b/2
        for idx, b in enumerate(lst):
            target = 3 * b / 2.0  # float target for a
            
            # Binary search in lst for the closest to target
            # We want a in lst such that a ≈ target
            lo, hi = 0, len(lst) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if lst[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid
            
            # Check lo and lo-1
            candidates = set()
            if lo < len(lst):
                candidates.add(lst[lo])
            if lo > 0:
                candidates.add(lst[lo - 1])
            
            for a in candidates:
                if a == b:
                    continue
                # Compute |a/b - 3/2| exactly
                # a/b - 3/2 = (2a - 3b) / (2b)
                num = abs(2 * a - 3 * b)
                den = 2 * b
                err = Fraction(num, den)
                if err < best_error:
                    best_error = err
                    best_pair = (a, b)
            
            # Check time
            if time.time() > deadline - 5:
                break
        
        if time.time() > deadline - 5:
            break
    
    a, b = best_pair
    return [a, b]

# EVOLVE-BLOCK-END
