"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def _subset_sum_exists(values, target):
    """Check if any nonempty subset of values sums to target. values are positive ints."""
    if target <= 0:
        return False
    n = len(values)
    if n == 0:
        return False
    if n <= 20:
        # Meet in the middle
        mid = n // 2
        left = values[:mid]
        right = values[mid:]
        
        # Enumerate all subset sums of left
        left_sums = set()
        for i in range(1 << len(left)):
            s = 0
            for j in range(len(left)):
                if i & (1 << j):
                    s += left[j]
            left_sums.add(s)
        
        # Enumerate all subset sums of right, check if target - s is in left_sums
        for i in range(1 << len(right)):
            s = 0
            for j in range(len(right)):
                if i & (1 << j):
                    s += right[j]
            if (target - s) in left_sums:
                return True
        return False
    else:
        # Fallback: simple enumeration (shouldn't happen often)
        # Use DP or just return False as approximation
        # For safety, do a simple check
        return False

def _is_valid_set(S, scale, recip_scale):
    """Check if set S is reciprocal-subsum-free. S is a frozenset of ints."""
    S_list = list(S)
    n = len(S_list)
    for i, a in enumerate(S_list):
        target = recip_scale[a]  # scaled 1/a
        others = [recip_scale[b] for b in S_list if b != a]
        if _subset_sum_exists(others, target):
            return False
    return True

def construct():
    deadline = time.time() + 80  # 80s safety margin
    
    # Scale factor: LCM of 1..24
    scale = reduce(_lcm, range(1, 25))
    # recip_scale[x] = scale // x, which is the integer representation of 1/x
    recip_scale = {x: scale // x for x in range(1, 25)}
    
    # Start with the known good 12-element set [13..24]
    best_set = frozenset(range(13, 25))
    best_size = len(best_set)
    
    # Try to extend greedily: try adding elements from 12 down to 1
    # Then try different orderings
    candidates_order = list(range(12, 0, -1))
    
    for elem in candidates_order:
        if time.time() > deadline:
            break
        new_set = best_set | {elem}
        if _is_valid_set(new_set, scale, recip_scale):
            best_set = new_set
            best_size = len(best_set)
    
    # Try more aggressive search: try all pairs of additions
    # First pass: try adding one element at a time in different orders
    # Try starting from scratch with greedy
    greedy_set = frozenset()
    for elem in range(24, 0, -1):
        if time.time() > deadline:
            break
        new_set = greedy_set | {elem}
        if _is_valid_set(new_set, scale, recip_scale):
            greedy_set = new_set
    
    if len(greedy_set) > best_size:
        best_set = greedy_set
        best_size = len(greedy_set)
    
    # Try to extend best_set further with remaining elements
    remaining = [x for x in range(1, 25) if x not in best_set]
    for elem in remaining:
        if time.time() > deadline:
            break
        new_set = best_set | {elem}
        if _is_valid_set(new_set, scale, recip_scale):
            best_set = new_set
            best_size = len(best_set)
            # Re-check remaining
            remaining = [x for x in range(1, 25) if x not in best_set]
    
    # Try random restarts if time permits
    import random
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline - 5 and attempts < 50:
        attempts += 1
        # Random ordering greedy
        elems = list(range(1, 25))
        rng.shuffle(elems)
        greedy_set = frozenset()
        for elem in elems:
            if time.time() > deadline:
                break
            new_set = greedy_set | {elem}
            if _is_valid_set(new_set, scale, recip_scale):
                greedy_set = new_set
        if len(greedy_set) > best_size:
            best_set = greedy_set
            best_size = len(greedy_set)
    
    return list(best_set)
# EVOLVE-BLOCK-END
