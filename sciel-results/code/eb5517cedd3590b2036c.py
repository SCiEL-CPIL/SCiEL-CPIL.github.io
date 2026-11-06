import random
from fractions import Fraction
import time

def construct():
    start_time = time.time()
    timeout = 55
    best_size = 0
    best_set = set()
    
    # Precompute reciprocals as Fractions
    recips = {i: Fraction(1, i) for i in range(1, 25)}
    
    # Run multiple randomized greedy attempts
    while time.time() - start_time < timeout:
        S = set()
        # Precompute subset sums for S? No, S changes.
        # We will maintain subset sums incrementally or recompute if small.
        # Given N=24, max size likely < 15.
        # Let's use a simple check that recomputes subset sums for the "others" set.
        
        candidates = list(range(1, 25))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() - start_time > timeout:
                break
            if x in S:
                continue
            if _can_add(S, x, recips):
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
            # If we found a very large set, we might stop early or try to improve
            # But let's keep running to see if we can do better.
            
    return sorted(best_set)

def _can_add(S, x, recips):
    """
    Check if x can be added to valid set S.
    S is a set of integers.
    recips is a dict {int: Fraction}.
    """
    if not S:
        return True
        
    # 1. Check if 1/x is a subset sum of S
    # Compute subset sums of S
    sums = {Fraction(0)}
    for y in S:
        new_sums = set()
        for s in sums:
            new_sums.add(s + recips[y])
        sums = sums | new_sums
        
    if recips[x] in sums:
        return False
        
    # 2. Check for each z in S if 1/z is a subset sum of (S \ {z}) U {x}
    # This is equivalent to checking if 1/z - 1/x is a subset sum of S \ {z}
    # Note: 1/z was NOT a subset sum of S \ {z} (since S is valid).
    # So we only need to check sums that include x.
    # Sum = 1/x + sum(T) where T subset of S \ {z}
    # So check if (1/z - 1/x) is a subset sum of S \ {z}
    
    for z in S:
        target_diff = recips[z] - recips[x]
        if target_diff < 0:
            # Subset sums are non-negative, so if target is negative, it can't be formed
            # Unless we allow negative coefficients? No, subsets are positive.
            # Wait, 1/z - 1/x can be negative if z > x.
            # If z > x, 1/z < 1/x, so 1/z - 1/x < 0.
            # Subset sums of positive numbers are >= 0.
            # So if target_diff < 0, it's impossible.
            continue
            
        # Compute subset sums of S \ {z}
        sums_z = {Fraction(0)}
        for y in S:
            if y == z:
                continue
            new_sums = set()
            for s in sums_z:
                new_sums.add(s + recips[y])
            sums_z = sums_z | new_sums
            
        if target_diff in sums_z:
            return False
            
    return True
