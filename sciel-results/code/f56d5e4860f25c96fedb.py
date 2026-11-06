"""Program for large reciprocal-subsum-free sets using optimized greedy + local search."""
import time
import math
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin under the 90s wall-clock limit


def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)


def _lcm(a, b):
    return a * b // math.gcd(a, b)


def _can_form_sum(target, items):
    """
    Check if target can be formed as a sum of a subset of items.
    Uses meet-in-the-middle for efficiency.
    """
    n = len(items)
    if n == 0:
        return target == 0
    if target == 0:
        return True
    if sum(items) < target:
        return False
    
    mid = n // 2
    left = items[:mid]
    right = items[mid:]
    
    left_sums = {0}
    for x in left:
        new_sums = set()
        for s in left_sums:
            ns = s + x
            if ns <= target:
                new_sums.add(ns)
        left_sums.update(new_sums)
        
    right_sums = {0}
    for x in right:
        new_sums = set()
        for s in right_sums:
            ns = s + x
            if ns <= target:
                new_sums.add(ns)
        right_sums.update(new_sums)
        
    for s in left_sums:
        if (target - s) in right_sums:
            return True
            
    return False


def _is_subsum_free(S):
    """
    Check if set S is reciprocal-subsum-free.
    For each x in S, 1/x must NOT equal sum(1/y for y in T)
    where T is a non-empty subset of S \\ {x}.
    """
    if len(S) < 2:
        return True
    
    S_list = list(S)
    n = len(S_list)
    
    L = 1
    for x in S_list:
        L = _lcm(L, x)
        
    vals = [L // x for x in S_list]
    
    for i in range(n):
        target = vals[i]
        others = [v for j, v in enumerate(vals) if j != i]
        
        if _can_form_sum(target, others):
            return False
            
    return True


def _try_add_elements(current, candidates, max_add=3):
    """
    Try to add up to max_add elements from candidates to current while maintaining subsum-freeness.
    Returns the best extended set found.
    """
    if not candidates or not current:
        return current
    
    best = set(current)
    
    # Try adding 1 element
    for c in candidates:
        if _time_remaining() < 3.0:
            break
        new_set = best | {c}
        if _is_subsum_free(new_set):
            if len(new_set) > len(best):
                best = new_set
    
    # Try adding 2 elements
    if len(candidates) >= 2 and _time_remaining() > 5.0:
        for c1, c2 in combinations(candidates, 2):
            if _time_remaining() < 3.0:
                break
            new_set = best | {c1, c2}
            if _is_subsum_free(new_set):
                if len(new_set) > len(best):
                    best = new_set
    
    # Try adding 3 elements
    if len(candidates) >= 3 and _time_remaining() > 8.0:
        for c1, c2, c3 in combinations(candidates, 3):
            if _time_remaining() < 3.0:
                break
            new_set = best | {c1, c2, c3}
            if _is_subsum_free(new_set):
                if len(new_set) > len(best):
                    best = new_set
    
    return best


def _swap_improve(current):
    """
    Try to improve current set by swapping out one element for one or more others.
    Returns the best set found.
    """
    if not current or _time_remaining() < 5.0:
        return current
    
    best = set(current)
    all_elements = set(range(1, 25))
    missing = all_elements - best
    
    if not missing:
        return best
    
    # Try swapping out one element for one or more missing elements
    for elem_to_remove in list(best):
        if _time_remaining() < 3.0:
            break
        
        base_without = best - {elem_to_remove}
        
        # Try adding one missing element
        for m in missing:
            if _time_remaining() < 2.0:
                break
            new_set = base_without | {m}
            if _is_subsum_free(new_set):
                if len(new_set) > len(best):
                    best = new_set
        
        # Try adding two missing elements
        if len(missing) >= 2 and _time_remaining() > 5.0:
            for m1, m2 in combinations(missing, 2):
                if _time_remaining() < 2.0:
                    break
                new_set = base_without | {m1, m2}
                if _is_subsum_free(new_set):
                    if len(new_set) > len(best):
                        best = new_set
    
    return best


def construct():
    """
    Return a nonempty subset of [1,24] that is reciprocal-subsum-free,
    maximizing cardinality.
    """
    current = set()
    
    # Greedy descending: try to add elements from 24 down to 1
    for k in range(24, 0, -1):
        if _time_remaining() < 5.0:
            break
        
        new_set = current | {k}
        if _is_subsum_free(new_set):
            current = new_set
    
    if not current:
        return [24]
    
    # Local search phase 1: try to add missing elements
    all_elements = set(range(1, 25))
    missing = all_elements - current
    
    if missing and _time_remaining() > 10.0:
        current = _try_add_elements(current, list(missing), max_add=3)
    
    # Local search phase 2: try swaps
    if _time_remaining() > 15.0:
        current = _swap_improve(current)
    
    # One more pass of adding if possible
    missing = all_elements - current
    if missing and _time_remaining() > 8.0:
        current = _try_add_elements(current, list(missing), max_add=2)
    
    return list(current)
