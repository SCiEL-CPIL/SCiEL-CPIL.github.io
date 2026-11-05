"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from math import gcd
from functools import reduce
import time
import sys

# EVOLVE-BLOCK-START
_DEADLINE = None
_BEST = None
_lcm_cache = {}

def _lcm(nums):
    """Compute LCM of a list of integers with caching."""
    key = tuple(sorted(nums))
    if key in _lcm_cache:
        return _lcm_cache[key]
    if not nums:
        return 1
    result = reduce(lambda a, b: a * b // gcd(a, b), nums)
    _lcm_cache[key] = result
    return result

def _check_safe(a, others):
    """
    Return True if no nonempty subset of `others` sums to 1/a.
    Uses meet-in-the-middle with integer arithmetic for speed.
    """
    n = len(others)
    if n == 0:
        return True
    
    # Scale all reciprocals by LCM of others and a
    lcm_val = _lcm(others + [a])
    
    # Target: 1/a * lcm_val = lcm_val / a
    target = lcm_val // a
    
    # Terms: 1/x * lcm_val = lcm_val // x for each x in others
    terms = [lcm_val // x for x in others]
    
    # All terms are positive integers. Sort descending for pruning.
    terms.sort(reverse=True)
    n = len(terms)
    
    # Quick check: if smallest term > target, no single element can match
    if terms[-1] > target:
        return True
    
    # Meet in the middle: split into two halves
    mid = n // 2
    left_terms = terms[:mid]
    right_terms = terms[mid:]
    
    # Generate all subset sums for left half (include 0 for empty subset)
    left_sums = set([0])
    for t in left_terms:
        new_sums = set()
        for s in left_sums:
            ns = s + t
            if ns <= target:
                new_sums.add(ns)
        left_sums.update(new_sums)
    
    # Check if target is achievable from left alone
    if target in left_sums:
        return False
    
    # Generate subset sums for right half and check
    right_sums = set([0])
    for t in right_terms:
        new_sums = set()
        for s in right_sums:
            ns = s + t
            if ns <= target:
                new_sums.add(ns)
        right_sums.update(new_sums)
    
    # Check all combinations
    for rs in right_sums:
        if target - rs in left_sums:
            # Need nonempty total subset. Since target > 0, at least one side is nonempty.
            if rs == 0 and (target - rs) == 0:
                continue
            return False
    
    return True

def _violates(a, S):
    """True if a in S violates: some subset of S\{a} sums to 1/a."""
    others = [x for x in S if x != a]
    return not _check_safe(a, others)

def _is_valid_set(S):
    """Check if the entire set S is valid."""
    for a in S:
        if _violates(a, S):
            return False
    return True

def _greedy_addition(order):
    """Greedy: try to add elements in given order, keep if valid."""
    S = []
    for a in order:
        if time.time() > _DEADLINE:
            break
        trial = S + [a]
        if _is_valid_set(trial):
            S = trial
    return S

def construct():
    global _DEADLINE, _BEST
    start = time.time()
    _DEADLINE = start + 85.0  # safety margin before 90s kill
    _lcm_cache.clear()

    full = list(range(1, 25))
    
    # Phase 1: Greedy addition, large to small
    # Larger elements have smaller reciprocals, less likely to form subsums
    S = _greedy_addition(sorted(full, reverse=True))
    _BEST = list(S)
    
    # Also try small to large for comparison
    S2 = _greedy_addition(sorted(full))
    if len(S2) > len(S):
        S = S2
    _BEST = list(S)
    
    # Try mixed orderings
    if time.time() < _DEADLINE:
        # Try: large, then small, then medium
        for order in [
            sorted(full, key=lambda x: -x),  # large to small
            sorted(full, key=lambda x: x),   # small to large
            sorted(full, key=lambda x: abs(x - 12)),  # center out
            sorted(full, key=lambda x: -x if x <= 12 else x),  # large small, then small large
        ]:
            if time.time() > _DEADLINE:
                break
            S3 = _greedy_addition(order)
            if len(S3) > len(S):
                S = S3
                _BEST = list(S)
    
    # Phase 2: Try to add back removed elements
    in_set = set(S)
    removed = [x for x in full if x not in in_set]
    for a in sorted(removed, reverse=True):
        if time.time() > _DEADLINE:
            break
        trial = S + [a]
        if _is_valid_set(trial):
            S = trial
    
    _BEST = list(S)
    
    # Phase 3: Local search (1-swap)
    improved = True
    while improved and time.time() < _DEADLINE:
        improved = False
        in_set = set(S)
        out_set = [x for x in full if x not in in_set]
        for a in out_set:
            if time.time() > _DEADLINE:
                break
            for b in list(S):
                if time.time() > _DEADLINE:
                    break
                trial = [x for x in S if x != b] + [a]
                if _is_valid_set(trial) and len(trial) > len(S):
                    S = trial
                    improved = True
                    break
            if improved:
                break
    
    _BEST = list(S)
    
    # Phase 4: Try to add pairs of out elements
    if time.time() < _DEADLINE:
        in_set = set(S)
        out_set = [x for x in full if x not in in_set]
        for i in range(len(out_set)):
            if time.time() > _DEADLINE:
                break
            for j in range(i + 1, len(out_set)):
                if time.time() > _DEADLINE:
                    break
                trial = S + [out_set[i], out_set[j]]
                if _is_valid_set(trial):
                    S = trial
                    in_set = set(S)
                    out_set = [x for x in full if x not in in_set]
                    break
            if time.time() > _DEADLINE:
                break
        
        # Try single additions again
        in_set = set(S)
        out_set = [x for x in full if x not in in_set]
        for a in out_set:
            if time.time() > _DEADLINE:
                break
            trial = S + [a]
            if _is_valid_set(trial):
                S = trial
    
    _BEST = list(S)
    
    return list(_BEST)
# EVOLVE-BLOCK-END
