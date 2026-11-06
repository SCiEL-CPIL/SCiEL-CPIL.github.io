"""Rewritten program for the reciprocal-subsum-free set task."""
import time
from fractions import Fraction
from itertools import combinations
from math import gcd

_START = None
DEADLINE = 50.0  # Safety margin before 90s kill

def _deadline_hit():
    global _START
    if _START is None:
        return False
    return (time.time() - _START) > DEADLINE


def _check_subset_sum(target_frac, candidates):
    """
    Check if target_frac (Fraction) can be formed by a non-empty subset sum
    of reciprocals of the given candidate integers.
    Uses integer arithmetic with LCM scaling for speed.
    """
    if not candidates:
        return False
    if target_frac <= 0:
        return False
    
    n = len(candidates)
    
    # Quick check: if target is larger than sum of all reciprocals, impossible
    total = sum(Fraction(1, c) for c in candidates)
    if target_frac > total:
        return False
    
    # Compute LCM of all candidates
    lcm_val = 1
    for c in candidates:
        lcm_val = lcm_val * c // gcd(lcm_val, c)
    
    target_scaled = target_frac * lcm_val
    if target_scaled.denominator != 1:
        return False
    target_int = target_scaled.numerator
    scaled_recs = [lcm_val // c for c in candidates]
    
    # For small n, brute force with early termination
    if n <= 15:
        # Brute force subset sum with pruning
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += scaled_recs[i]
                    if s > target_int:
                        break
            else:
                if s == target_int:
                    return True
        return False
    else:
        # Meet-in-the-middle for larger n using integer arithmetic
        half = n // 2
        left = scaled_recs[:half]
        right = scaled_recs[half:]
        
        # Precompute all subset sums for left half
        left_sums = set()
        for mask in range(1 << len(left)):
            s = 0
            for i in range(len(left)):
                if mask & (1 << i):
                    s += left[i]
            left_sums.add(s)
        
        # Check right half subsets against left sums
        for mask in range(1 << len(right)):
            s = 0
            for i in range(len(right)):
                if mask & (1 << i):
                    s += right[i]
            if (target_int - s) in left_sums:
                # Need to ensure we're using a non-empty subset overall
                # If mask==0 and target_int in left_sums, that's fine (left non-empty)
                # If mask!=0, right is non-empty, so overall is non-empty
                # Only issue: mask==0 AND target_int==0, but target_int>0 guaranteed
                return True
        return False


def _is_valid_set(S):
    """
    Check if set S (list of ints) is reciprocal-subsum-free.
    For each u in S, 1/u must not be a subset sum of {1/w : w in S, w != u}.
    """
    if len(S) < 2:
        return True
    
    S_sorted = sorted(S)
    for i, u in enumerate(S_sorted):
        target = Fraction(1, u)
        # Candidates are all other elements
        candidates = [w for j, w in enumerate(S_sorted) if j != i]
        if _check_subset_sum(target, candidates):
            return False
    return True


def _is_valid_addition(S, new_elem):
    """
    Check if adding new_elem to valid set S keeps the set valid.
    Only need to check:
    1. new_elem's reciprocal vs others in S
    2. Each existing element's reciprocal vs S + new_elem
    """
    if not S:
        return True
    
    # Check new_elem's reciprocal against all others
    target = Fraction(1, new_elem)
    if _check_subset_sum(target, S):
        return False
    
    # Check each existing element's reciprocal against S + new_elem
    for u in S:
        target_u = Fraction(1, u)
        candidates = [w for w in S if w != u] + [new_elem]
        if _check_subset_sum(target_u, candidates):
            return False
    
    return True


def _greedy_construct():
    """
    Greedy construction: add elements one by one, keeping the set valid.
    Try different orderings to find a good starting point.
    Uses incremental validity check for speed.
    """
    best_set = []
    best_size = 0
    
    # Try adding in increasing order
    S = []
    for v in range(1, 25):
        if _deadline_hit():
            break
        if _is_valid_addition(S, v):
            S.append(v)
    
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]
    
    # Try adding in decreasing order
    S = []
    for v in range(24, 0, -1):
        if _deadline_hit():
            break
        if _is_valid_addition(S, v):
            S.append(v)
    
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]
    
    # Try skipping 1 (since 1/1 = 1 is hard to avoid)
    S = []
    for v in range(2, 25):
        if _deadline_hit():
            break
        if _is_valid_addition(S, v):
            S.append(v)
    
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]
    
    # Try a mixed order: larger numbers first (safer), then smaller
    S = []
    for v in range(24, 1, -1):
        if _deadline_hit():
            break
        if _is_valid_addition(S, v):
            S.append(v)
    
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]
    
    # Try evens first, then odds (evens have smaller reciprocals, less likely to form subsums)
    S = []
    for v in [24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1]:
        if _deadline_hit():
            break
        if _is_valid_addition(S, v):
            S.append(v)
    
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]
    
    # Try odds first, then evens
    S = []
    for v in [23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2]:
        if _deadline_hit():
            break
        if _is_valid_addition(S, v):
            S.append(v)
    
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]
    
    return best_set


def _local_search(initial_set):
    """
    Local search: try to improve the set by adding elements or swapping.
    """
    S = set(initial_set)
    best_set = list(S)
    best_size = len(S)
    
    improved = True
    while improved and not _deadline_hit():
        improved = False
        
        # Try adding one element
        for add in range(1, 25):
            if _deadline_hit():
                break
            if add in S:
                continue
            if _is_valid_addition(list(S), add):
                S.add(add)
                best_set = list(S)
                best_size = len(S)
                improved = True
        
        # Try removing one element and adding another (to potentially grow)
        for remove in list(S):
            if _deadline_hit():
                break
            S_without = S - {remove}
            for add in range(1, 25):
                if add in S:
                    continue
                if _is_valid_addition(list(S_without), add):
                    new_S = S_without | {add}
                    if len(new_S) > best_size:
                        S = new_S
                        best_set = list(S)
                        best_size = len(S)
                        improved = True
                        break
            if improved:
                break
    
    return best_set


def _branch_and_bound():
    """
    Branch and bound search with strong pruning.
    Uses incremental validity check.
    """
    candidates = list(range(1, 25))
    best_set = []
    best_size = 0
    
    def backtrack(idx, S):
        nonlocal best_set, best_size
        
        if _deadline_hit():
            return
        
        # Pruning
        if len(S) + (len(candidates) - idx) <= best_size:
            return
        
        if idx == len(candidates):
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
            return
        
        v = candidates[idx]
        
        # Option 1: Include v
        if _is_valid_addition(S, v):
            backtrack(idx + 1, S + [v])
        
        # Option 2: Exclude v
        backtrack(idx + 1, S)
    
    backtrack(0, [])
    return best_set


def _conflict_graph_approach():
    """
    Try multiple random orderings with incremental validity check.
    """
    import random
    best_set = []
    best_size = 0
    
    for trial in range(30):
        if _deadline_hit():
            break
        
        order = list(range(1, 25))
        random.shuffle(order)
        
        S = []
        for v in order:
            if _deadline_hit():
                break
            if _is_valid_addition(S, v):
                S.append(v)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
    
    return best_set


def construct():
    global _START
    _START = time.time()
    
    # Start with greedy
    result = _greedy_construct()
    
    # Improve with local search
    result = _local_search(result)
    
    # Try conflict graph approach if time permits
    if not _deadline_hit():
        cg_result = _conflict_graph_approach()
        if len(cg_result) > len(result):
            result = cg_result
    
    # Try branch and bound if time permits
    if not _deadline_hit():
        bb_result = _branch_and_bound()
        if len(bb_result) > len(result):
            result = bb_result
    
    # Final local search polish
    if not _deadline_hit():
        result = _local_search(result)
    
    # Safety: ensure nonempty
    if not result:
        result = [24]
    
    return result
