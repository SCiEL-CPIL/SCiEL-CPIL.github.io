import math
import random
import time

# Precompute LCM of 1..24
def _compute_lcm(n):
    result = 1
    for i in range(1, n + 1):
        result = math.lcm(result, i)
    return result

_M = _compute_lcm(24)
_v = [0] * 25
for _i in range(1, 25):
    _v[_i] = _M // _i

def _is_subset_sum(target, values):
    """
    Check if target can be formed by a subset of values using meet-in-the-middle.
    """
    n = len(values)
    if n == 0:
        return target == 0
    
    mid = n // 2
    left = values[:mid]
    right = values[mid:]
    
    # Generate subset sums for left
    left_sums = {0}
    for val in left:
        # Create new sums by adding val to existing sums
        new_sums = set()
        for s in left_sums:
            new_sums.add(s + val)
        left_sums.update(new_sums)
        
    # Generate subset sums for right
    right_sums = {0}
    for val in right:
        new_sums = set()
        for s in right_sums:
            new_sums.add(s + val)
        right_sums.update(new_sums)
    
    # Check if any pair sums to target
    for ls in left_sums:
        if (target - ls) in right_sums:
            return True
            
    return False

def _is_valid(S):
    """
    Check if S is reciprocal-subsum-free.
    S: set of integers.
    """
    if not S:
        return True
        
    for x in S:
        others = [_v[y] for y in S if y != x]
        if _is_subset_sum(_v[x], others):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Initial set: 13 to 24
    S = set(range(13, 25))
    
    if not _is_valid(S):
        return [24]
        
    best_S = set(S)
    best_size = len(best_S)
    
    # Try to add elements from 1 to 12
    # We do multiple passes with different orderings to escape local optima
    for _ in range(100):
        if time.time() > deadline:
            break
            
        # Random restart logic
        S_curr = set(range(13, 25))
        
        candidates = list(range(1, 13))
        random.shuffle(candidates)
        
        for c in candidates:
            if time.time() > deadline:
                break
            new_S = S_curr | {c}
            if _is_valid(new_S):
                S_curr = new_S
        
        if len(S_curr) > best_size:
            best_size = len(S_curr)
            best_S = set(S_curr)
            
        # If we found a set of size 24, we can stop
        if best_size == 24:
            break

    return list(best_S)
