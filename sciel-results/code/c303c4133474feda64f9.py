"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
import time
import itertools
import sys

# EVOLVE-BLOCK-START
_DEADLINE = None
_BEST = None

def _can_form_target(target, terms, deadline=None):
    """Return True if some nonempty subset of terms sums exactly to target.
    terms is a list of Fractions (the reciprocals)."""
    n = len(terms)
    if n == 0:
        return False
    
    # Sort terms descending for better pruning (larger terms first)
    terms_sorted = sorted(terms, reverse=True)
    
    # Precompute suffix sums for pruning
    suffix_sums = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + terms_sorted[i]
    
    # If total sum < target, impossible
    if suffix_sums[0] < target:
        return False
    
    # Use memoization: state = (index, current_sum)
    # But current_sum is Fraction, so use a set of (i, acc) tuples
    visited = set()
    
    def dfs(i, acc):
        if acc == target:
            return True
        if acc > target:
            return False
        if i == n:
            return False
        # Pruning: even taking all remaining can't reach target
        if acc + suffix_sums[i] < target:
            return False
        
        state = (i, acc)
        if state in visited:
            return False
        visited.add(state)
        
        # Try including term i
        if dfs(i + 1, acc + terms_sorted[i]):
            return True
        # Try excluding term i
        if dfs(i + 1, acc):
            return True
        return False
    
    return dfs(0, Fraction(0))

def _is_safe_decreasing(a, larger_elements):
    """Check if 1/a can be formed by a subset of reciprocals of larger_elements.
    larger_elements should all be > a."""
    if not larger_elements:
        return True
    target = Fraction(1, a)
    # Reciprocals of larger elements are all < 1/a (since x > a means 1/x < 1/a)
    # So we need to check if any subset sums to exactly 1/a
    terms = [Fraction(1, x) for x in larger_elements]
    return not _can_form_target(target, terms)

def _greedy_decreasing():
    """Build a maximal set by adding elements from 24 down to 1.
    When adding element a, only check against elements already in the set (which are all > a)."""
    S = []
    for a in range(24, 0, -1):
        # Check if 1/a can be formed by subset of reciprocals of elements already in S
        # All elements in S are > a, so their reciprocals are < 1/a
        if _is_safe_decreasing(a, S):
            S.append(a)
    return S

def _full_validate(S):
    """Full validation: for each a in S, check no subset of S\{a} sums to 1/a."""
    for a in S:
        others = [x for x in S if x != a]
        if not others:
            continue
        target = Fraction(1, a)
        terms = [Fraction(1, x) for x in others]
        if _can_form_target(target, terms):
            return False
    return True

def construct():
    global _DEADLINE, _BEST
    start = time.time()
    _DEADLINE = start + 80.0  # safety margin before 90s kill

    # Phase 1: Fast greedy decreasing construction
    S = _greedy_decreasing()
    _BEST = list(S)
    
    # Phase 2: Verify the greedy result is actually valid
    # (it should be by construction, but double-check)
    if not _full_validate(S):
        # Fallback: remove violators
        changed = True
        while changed:
            if time.time() > _DEADLINE:
                break
            changed = False
            for a in sorted(S):
                if time.time() > _DEADLINE:
                    break
                others = [x for x in S if x != a]
                if others:
                    target = Fraction(1, a)
                    terms = [Fraction(1, x) for x in others]
                    if _can_form_target(target, terms):
                        S.remove(a)
                        changed = True
                        break
        _BEST = list(S)
    
    # Phase 3: Try to improve by swapping elements
    # We want to find a valid set of size > len(S)
    # Strategy: try removing one element and adding one or more others
    
    current_size = len(S)
    in_set = set(S)
    out_set = [x for x in range(1, 25) if x not in in_set]
    
    # Try 1-for-1 swaps that might enable adding more elements
    improved = True
    while improved and time.time() < _DEADLINE:
        improved = False
        S_list = list(S)
        for a in out_set:
            if time.time() > _DEADLINE:
                break
            for b in S_list:
                if time.time() > _DEADLINE:
                    break
                trial = [x for x in S_list if x != b] + [a]
                if _full_validate(trial):
                    # Same size, but might enable further improvements
                    # Only accept if it leads to a different configuration
                    # For now, skip same-size swaps unless they enable growth
                    pass
        
        # Try adding elements one by one to see if we can grow
        for a in out_set:
            if time.time() > _DEADLINE:
                break
            trial = list(S) + [a]
            if _full_validate(trial):
                S = trial
                current_size = len(S)
                in_set = set(S)
                out_set = [x for x in range(1, 25) if x not in in_set]
                _BEST = list(S)
                improved = True
                break
    
    # Phase 4: More aggressive search - try to find larger valid sets
    # Use a different greedy order or random restarts if time permits
    if time.time() < _DEADLINE - 10:
        # Try several different greedy orders
        import random
        best_found = list(S)
        best_size = len(S)
        
        # Try greedy from different starting points
        for trial_start in range(24, 0, -1):
            if time.time() > _DEADLINE - 5:
                break
            
            # Create a permutation starting with trial_start
            remaining = [x for x in range(1, 25) if x != trial_start]
            random.shuffle(remaining)
            order = [trial_start] + remaining
            
            trial_S = []
            for a in order:
                if _is_safe_decreasing(a, [x for x in trial_S if x > a]):
                    # Also need to check against smaller elements already in trial_S
                    # For simplicity, do full check
                    if _full_validate(trial_S + [a]):
                        trial_S.append(a)
            
            if len(trial_S) > best_size:
                best_size = len(trial_S)
                best_found = trial_S
                _BEST = list(trial_S)
        
        S = best_found
        _BEST = list(S)
    
    return list(_BEST)
# EVOLVE-BLOCK-END
