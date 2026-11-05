"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
import time
import itertools
import sys
import math
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START
_DEADLINE = None
_BEST = None

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    return reduce(_lcm, lst)

def _meet_in_middle_check(target, terms):
    """Check if any subset of terms sums to target using meet-in-the-middle.
    terms and target are integers (scaled reciprocals)."""
    n = len(terms)
    if n == 0:
        return False
    if sum(terms) < target:
        return False
    if target == 0:
        return False
    
    half = n // 2
    left = terms[:half]
    right = terms[half:]
    
    # Generate all subset sums for left half
    left_sums = set()
    for mask in range(1 << len(left)):
        s = 0
        for i in range(len(left)):
            if mask & (1 << i):
                s += left[i]
        left_sums.add(s)
    
    # For each subset sum of right half, check if target - right_sum is in left_sums
    for mask in range(1 << len(right)):
        s = 0
        for i in range(len(right)):
            if mask & (1 << i):
                s += right[i]
        if (target - s) in left_sums:
            return True
    
    return False

def _can_form_target_int(target_int, terms_int):
    """Check if any nonempty subset of terms_int sums to target_int.
    All values are positive integers."""
    if not terms_int:
        return False
    total = sum(terms_int)
    if total < target_int:
        return False
    return _meet_in_middle_check(target_int, terms_int)

def _is_safe_decreasing_int(a, larger_elements, lcm_val):
    """Check if 1/a can be formed by subset of reciprocals of larger_elements.
    Uses integer arithmetic with scaling by lcm_val."""
    if not larger_elements:
        return True
    target_int = lcm_val // a
    terms_int = [lcm_val // x for x in larger_elements]
    return not _can_form_target_int(target_int, terms_int)

def _full_validate_int(S):
    """Full validation using integer arithmetic."""
    if not S:
        return True
    lcm_val = _lcm_list(S)
    for a in S:
        others = [x for x in S if x != a]
        if not others:
            continue
        target_int = lcm_val // a
        terms_int = [lcm_val // x for x in others]
        if _can_form_target_int(target_int, terms_int):
            return False
    return True

def _greedy_decreasing_int():
    """Build a maximal set using integer arithmetic for speed."""
    S = []
    # Precompute lcm as we go
    for a in range(24, 0, -1):
        if not S:
            S.append(a)
            continue
        # Check against existing elements (all > a)
        if _is_safe_decreasing_int(a, S, _lcm_list(S)):
            S.append(a)
    return S

def _try_add_element(S, a, lcm_val=None):
    """Try to add element a to set S, returning new set if valid."""
    if a in S:
        return None
    if lcm_val is None:
        lcm_val = _lcm_list(S) if S else 1
    trial = S + [a]
    if _full_validate_int(trial):
        return trial
    return None

def construct():
    global _DEADLINE, _BEST
    start = time.time()
    _DEADLINE = start + 80.0  # safety margin before 90s kill

    # Phase 1: Fast greedy decreasing construction with integer arithmetic
    S = _greedy_decreasing_int()
    _BEST = list(S)
    
    # Phase 2: Verify the greedy result
    if not _full_validate_int(S):
        changed = True
        while changed and time.time() < _DEADLINE:
            changed = False
            for a in sorted(S):
                if time.time() > _DEADLINE:
                    break
                others = [x for x in S if x != a]
                if others:
                    lcm_val = _lcm_list(S)
                    target_int = lcm_val // a
                    terms_int = [lcm_val // x for x in others]
                    if _can_form_target_int(target_int, terms_int):
                        S.remove(a)
                        changed = True
                        break
        _BEST = list(S)
    
    # Phase 3: Try to add more elements
    current_size = len(S)
    in_set = set(S)
    out_set = [x for x in range(1, 25) if x not in in_set]
    
    improved = True
    while improved and time.time() < _DEADLINE:
        improved = False
        for a in out_set:
            if time.time() > _DEADLINE:
                break
            trial = _try_add_element(S, a)
            if trial is not None:
                S = trial
                current_size = len(S)
                in_set = set(S)
                out_set = [x for x in range(1, 25) if x not in in_set]
                _BEST = list(S)
                improved = True
                break
    
    # Phase 4: Try 1-for-1 swaps to enable further growth
    if time.time() < _DEADLINE - 15:
        for _ in range(50):  # Limited number of swap attempts
            if time.time() > _DEADLINE - 5:
                break
            found_swap = False
            for a in out_set:
                if time.time() > _DEADLINE - 5:
                    break
                for b in list(S):
                    if time.time() > _DEADLINE - 5:
                        break
                    trial = [x for x in S if x != b] + [a]
                    if _full_validate_int(trial):
                        # Check if this enables adding more elements
                        in_set_trial = set(trial)
                        out_set_trial = [x for x in range(1, 25) if x not in in_set_trial]
                        best_trial = list(trial)
                        for c in out_set_trial:
                            if time.time() > _DEADLINE - 5:
                                break
                            trial2 = _try_add_element(trial, c)
                            if trial2 is not None and len(trial2) > len(best_trial):
                                best_trial = trial2
                        if len(best_trial) > len(S):
                            S = best_trial
                            _BEST = list(S)
                            in_set = set(S)
                            out_set = [x for x in range(1, 25) if x not in in_set]
                            found_swap = True
                            break
                if found_swap:
                    break
            if not found_swap:
                break
    
    # Phase 5: Random restarts with different orders if time permits
    if time.time() < _DEADLINE - 20:
        import random
        best_found = list(S)
        best_size = len(S)
        
        for _ in range(20):  # Limited random restarts
            if time.time() > _DEADLINE - 10:
                break
            
            order = list(range(1, 25))
            random.shuffle(order)
            
            trial_S = []
            for a in order:
                if time.time() > _DEADLINE - 10:
                    break
                if not trial_S:
                    trial_S.append(a)
                    continue
                # Check if a can be safely added
                if _is_safe_decreasing_int(a, [x for x in trial_S if x > a], _lcm_list(trial_S)):
                    # Also need full check for correctness
                    if _full_validate_int(trial_S + [a]):
                        trial_S.append(a)
            
            if len(trial_S) > best_size:
                best_size = len(trial_S)
                best_found = trial_S
                _BEST = list(trial_S)
        
        S = best_found
        _BEST = list(S)
    
    return list(_BEST)
# EVOLVE-BLOCK-END
