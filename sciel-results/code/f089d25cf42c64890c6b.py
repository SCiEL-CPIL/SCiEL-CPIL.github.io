"""
Program for large reciprocal-subsum-free sets.
Uses integer arithmetic with LCM scaling for fast subset-sum checks.
Implements greedy construction with multiple random restarts and local search.
"""
import time
import math
import random
from math import gcd

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin under the 90s wall-clock limit


def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)


def _lcm(a, b):
    """Compute LCM of two integers."""
    return a * b // gcd(a, b)


def _lcm_list(lst):
    """Compute LCM of a list of integers."""
    if not lst:
        return 1
    result = 1
    for x in lst:
        result = _lcm(result, x)
    return result


def _can_form_sum(target, items):
    """
    Check if target can be formed as a sum of a non-empty subset of items.
    All values are positive integers.
    Uses DP with a set of achievable sums, pruning sums > target.
    """
    if not items:
        return False
    if target <= 0:
        return False
    
    # Quick check: sum of all items
    total = sum(items)
    if total < target:
        return False
    
    # If target equals any single item, return True
    if target in items:
        return True
    
    # DP: set of achievable sums
    achievable = {0}
    for item in items:
        new_sums = set()
        for s in achievable:
            ns = s + item
            if ns <= target:
                new_sums.add(ns)
        achievable.update(new_sums)
        if target in achievable:
            return True
    
    return target in achievable


def _is_subsum_free(selected, candidate):
    """
    Check if adding candidate to selected keeps the set subsum-free.
    selected: list of ints
    candidate: int
    """
    # Check for candidate: 1/candidate should not equal sum of reciprocals of any non-empty subset of selected
    if selected:
        all_items = selected + [candidate]
        lcm_val = _lcm_list(all_items)
        target_int = lcm_val // candidate
        items_int = [lcm_val // x for x in selected]
        if _can_form_sum(target_int, items_int):
            return False
    
    # Check for each element in selected: 1/x should not equal sum of reciprocals of any non-empty subset of (selected - {x}) + {candidate}
    for x in selected:
        others = [y for y in selected if y != x]
        others.append(candidate)
        lcm_val = _lcm_list(others)
        target_int = lcm_val // x
        items_int = [lcm_val // y for y in others]
        if _can_form_sum(target_int, items_int):
            return False
            
    return True


def _validate_set(s):
    """Validate that a set is subsum-free."""
    for x in s:
        others = [y for y in s if y != x]
        if not others:
            continue
        lcm_val = _lcm_list(others)
        target_int = lcm_val // x
        items_int = [lcm_val // y for y in others]
        if _can_form_sum(target_int, items_int):
            return False
    return True


def _greedy_construct(order):
    """Greedy construction given an ordering of candidates."""
    trial_set = []
    for candidate in order:
        if _time_remaining() < 2.0:
            break
        if _is_subsum_free(trial_set, candidate):
            trial_set.append(candidate)
    return trial_set


def _local_improve(s):
    """Try to improve the set by adding elements or swapping."""
    s = list(s)
    all_elements = set(range(1, 25))
    current_set = set(s)
    
    # Try to add elements
    improved = True
    while improved and _time_remaining() > 3.0:
        improved = False
        for candidate in sorted(all_elements - current_set, reverse=True):
            if _is_subsum_free(s, candidate):
                s.append(candidate)
                current_set.add(candidate)
                improved = True
                break
    
    # Try swaps: remove one, try to add two
    for i in range(len(s)):
        if _time_remaining() < 3.0:
            break
        removed = s[i]
        remaining = s[:i] + s[i+1:]
        remaining_set = set(remaining)
        
        # Try to add two elements to replace one
        for add1 in sorted(all_elements - remaining_set, reverse=True):
            if _time_remaining() < 3.0:
                break
            trial = remaining + [add1]
            trial_set = set(trial)
            for add2 in sorted(all_elements - trial_set, reverse=True):
                if _time_remaining() < 3.0:
                    break
                trial2 = trial + [add2]
                if _validate_set(trial2) and len(trial2) > len(s):
                    s = trial2
                    current_set = set(s)
                    return s
    
    return s


def construct():
    """
    Construct a large reciprocal-subsum-free subset of [1, 24].
    """
    all_elements = list(range(1, 25))
    best_set = []
    
    # 1. Deterministic greedy: Large to Small
    if _time_remaining() > 5.0:
        s = _greedy_construct(sorted(all_elements, reverse=True))
        s = _local_improve(s)
        if len(s) > len(best_set):
            best_set = s

    # 2. Deterministic greedy: Small to Large
    if _time_remaining() > 5.0:
        s = _greedy_construct(sorted(all_elements))
        s = _local_improve(s)
        if len(s) > len(best_set):
            best_set = s
            
    # 3. Random restarts with local improvement
    random.seed(42)
    
    for _ in range(300):
        if _time_remaining() < 10.0:
            break
            
        # Random order
        candidates = list(all_elements)
        random.shuffle(candidates)
        
        trial_set = _greedy_construct(candidates)
        trial_set = _local_improve(trial_set)
        
        if len(trial_set) > len(best_set):
            best_set = trial_set
            
    # 4. Biased random restarts: favor larger elements
    for _ in range(100):
        if _time_remaining() < 10.0:
            break
            
        # Bias toward larger elements
        candidates = list(all_elements)
        # Shuffle with bias: larger elements more likely to come first
        weighted = []
        for i, c in enumerate(candidates):
            weight = c / 24.0  # Larger elements get higher weight
            weighted.append((weight, c))
        random.shuffle(weighted)
        order = [c for _, c in weighted]
        
        trial_set = _greedy_construct(order)
        trial_set = _local_improve(trial_set)
        
        if len(trial_set) > len(best_set):
            best_set = trial_set
            
    # Final validation
    if best_set and not _validate_set(best_set):
        # Fallback to a known valid set
        best_set = [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13]
    
    return sorted(best_set)

if __name__ == "__main__":
    s = construct()
    print(s)
    print(len(s))
