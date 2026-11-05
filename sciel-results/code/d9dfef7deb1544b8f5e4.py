from fractions import Fraction
import time
import itertools
from math import gcd

_DEADLINE = None
_BEST = None

def _lcm(a, b):
    return a * b // gcd(a, b)

def _subset_sum_exists_int(target_int, elements_int):
    """
    Check if target_int can be formed by a subset of elements_int.
    elements_int are positive integers.
    """
    if not elements_int:
        return False
    elements_int = sorted(elements_int, reverse=True)
    total = sum(elements_int)
    if total < target_int:
        return False
    
    n = len(elements_int)
    suffix_sums = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + elements_int[i]
        
    def dfs_opt(idx, acc):
        if acc == target_int:
            return True
        if idx == n:
            return False
        if acc > target_int:
            return False
        if acc + suffix_sums[idx] < target_int:
            return False
            
        if dfs_opt(idx + 1, acc + elements_int[idx]):
            return True
        if dfs_opt(idx + 1, acc):
            return True
        return False
        
    return dfs_opt(0, 0)

def _init():
    global _DEADLINE, _BEST
    _DEADLINE = time.time() + 50.0
    _BEST = None

def _subset_sum_exists(target_frac, elements):
    """
    Check if target_frac can be formed by a non-empty subset of reciprocals of elements.
    """
    if not elements:
        return False
        
    x = target_frac.denominator
    l = x
    for e in elements:
        l = _lcm(l, e)
        
    target_int = l // x
    elements_int = [l // e for e in elements]
    
    return _subset_sum_exists_int(target_int, elements_int)

def _greedy_decreasing():
    """
    Build set by adding elements from 24 down to 1.
    """
    selected = []
    for x in range(24, 0, -1):
        if _subset_sum_exists(Fraction(1, x), selected):
            continue
        else:
            selected.append(x)
    return selected

def _local_search_1_out_k_in(S):
    """
    Try to improve S by removing one element and adding multiple candidates.
    """
    S = list(S)
    improved = True
    while improved and time.time() < _DEADLINE:
        improved = False
        S_set = set(S)
        B = [x for x in range(1, 25) if x not in S_set]
        
        # Try removing each y in S
        for y in S:
            S_temp = [x for x in S if x != y]
            S_temp_set = set(S_temp)
            
            candidates = []
            for x in B:
                # Check if x is valid with respect to S_temp
                if not _subset_sum_exists(Fraction(1, x), S_temp):
                    candidates.append(x)
                    
            if len(candidates) > 1:
                # We can add all candidates
                # Verify validity of the new set just in case (should be valid by proof)
                new_S = S_temp + candidates
                # Optional: Full validation check for safety
                # if _is_valid_set(new_S):
                S = new_S
                improved = True
                break # Restart the while loop with the new S
        
    return S

def _is_valid_set(S):
    """Check if S is a valid reciprocal-subsum-free set."""
    S_list = sorted(S, reverse=True)
    for x in S_list:
        target = Fraction(1, x)
        others = [y for y in S_list if y != x]
        if _subset_sum_exists(target, others):
            return False
    return True

def construct():
    _init()
    
    # Start with decreasing greedy
    best = _greedy_decreasing()
    
    if not best:
        best = [24]
        
    # Apply local search
    best = _local_search_1_out_k_in(best)
    
    # Final check and return
    if _is_valid_set(best):
        _BEST = best
    else:
        # Fallback to greedy if local search broke it (should not happen)
        _BEST = _greedy_decreasing()
        
    return list(_BEST)
