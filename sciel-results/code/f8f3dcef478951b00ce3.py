import time
from math import lcm
import bisect
import random

_START_TIME = time.time()
_DEADLINE = 45.0

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

# Precompute LCM of 1..24
_L = 1
for i in range(1, 25):
    _L = lcm(_L, i)

# Precompute integer reciprocals: R[x] = _L // x
_R = [0] * 25
for i in range(1, 25):
    _R[i] = _L // i

def _subset_sum_exists(target, elements):
    """
    Check if target can be formed by a non-empty subset of elements.
    elements is a list of integers (scaled reciprocals).
    target is an integer (scaled reciprocal).
    Uses Meet-in-the-Middle.
    """
    if not elements:
        return False
    if target <= 0:
        return False
    
    n = len(elements)
    
    # If n is small, simple set DP is fine and faster overhead
    if n <= 12:
        reachable = {0}
        for e in elements:
            new_reachable = set()
            for s in reachable:
                if s <= target:
                    new_reachable.add(s)
                if s + e <= target:
                    new_reachable.add(s + e)
            reachable = new_reachable
        return target in reachable
    
    # Meet in the middle
    mid = n // 2
    left = elements[:mid]
    right = elements[mid:]
    
    # Generate subset sums for left
    sums_left = [0]
    for e in left:
        new_sums = [s + e for s in sums_left if s + e <= target]
        sums_left.extend(new_sums)
    
    # Generate subset sums for right
    sums_right = [0]
    for e in right:
        new_sums = [s + e for s in sums_right if s + e <= target]
        sums_right.extend(new_sums)
        
    # Sort right sums for binary search
    sums_right.sort()
    
    # Check for match
    for s_l in sums_left:
        if s_l > target:
            continue
        needed = target - s_l
        # Binary search needed in sums_right
        idx = bisect.bisect_left(sums_right, needed)
        if idx < len(sums_right) and sums_right[idx] == needed:
            # Ensure subset is not empty? 
            # If target > 0, and s_l + s_r = target, then at least one sum is > 0.
            # Since elements are positive, sum > 0 implies non-empty subset.
            return True
            
    return False

def _is_valid(S):
    """
    Check if S is a valid reciprocal-subsum-free set.
    S is a list of integers.
    """
    for k in S:
        target = _R[k]
        others = [_R[x] for x in S if x != k]
        if _subset_sum_exists(target, others):
            return False
    return True

def _greedy_with_order(order):
    """Greedy construction using a specific element order."""
    S = []
    for k in order:
        if _is_valid(S + [k]):
            S.append(k)
    return S

def _local_search(S, deadline):
    """Try to improve set S by swap operations (remove one, add another)."""
    S = sorted(S)
    improved = True
    while improved and _time_remaining() > 3:
        improved = False
        for i in range(len(S)):
            for j in range(1, 25):
                if j in S:
                    continue
                new_S = S[:i] + S[i+1:] + [j]
                if _is_valid(new_S):
                    S = sorted(new_S)
                    improved = True
                    break
            if improved:
                break
    return S

def construct():
    # Multiple greedy orders to find a good initial solution
    # "Safest" elements are large numbers (small reciprocals)
    orders = [
        list(range(24, 0, -1)),  # descending
        list(range(1, 25)),      # ascending
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        # Some random-ish orders
        [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [24, 1, 23, 2, 22, 3, 21, 4, 20, 5, 19, 6, 18, 7, 17, 8, 16, 9, 15, 10, 14, 11, 13, 12],
    ]
    
    best_greedy = []
    for order in orders:
        if _time_remaining() < 5:
            break
        S_g = _greedy_with_order(order)
        if len(S_g) > len(best_greedy):
            best_greedy = S_g
    
    # Local search to try to improve
    if _time_remaining() < 3:
        best_set = best_greedy
    else:
        best_set = _local_search(best_greedy, _DEADLINE)
    
    # If we have time, try to improve further with more local search iterations
    # or try adding elements greedily to the current best set
    if _time_remaining() > 5:
        # Try to add any missing element to the best set
        improved = True
        while improved and _time_remaining() > 5:
            improved = False
            for j in range(1, 25):
                if j not in best_set:
                    new_S = best_set + [j]
                    if _is_valid(new_S):
                        best_set = new_S
                        improved = True
                        break
        
        # Run local search again on the potentially larger set
        if improved or _time_remaining() > 10:
            best_set = _local_search(best_set, _DEADLINE)
            
    if not best_set:
        return list(range(13, 25))
        
    return sorted(best_set)

if __name__ == "__main__":
    s = construct()
    print(s)
    print(len(s))
