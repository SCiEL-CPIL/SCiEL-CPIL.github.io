"""Rewritten program: branch-and-bound search for max-cardinality reciprocal-subsum-free subset of [1,24]."""
import time
from math import gcd
from functools import reduce
from itertools import combinations

# EVOLVE-BLOCK-START
def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(vals):
    if not vals:
        return 1
    return reduce(_lcm, vals)

def _subset_sum_exists(target, vals):
    """Return True if some subset of vals sums to target (exact integer)."""
    if target == 0:
        return True
    if not vals:
        return False
    # Filter: any val > target can't be used (all positive)
    vals = [v for v in vals if v <= target]
    if not vals:
        return False
    if sum(vals) < target:
        return False
    n = len(vals)
    if n <= 20:
        # Direct meet-in-the-middle
        half = n // 2
        left = vals[:half]
        right = vals[half:]
        left_sums = set()
        # generate all subset sums of left
        left_sums.add(0)
        for v in left:
            left_sums |= {s + v for s in left_sums}
        # for each right subset sum, check if target - rs in left_sums
        right_sums = {0}
        for v in right:
            right_sums |= {s + v for s in right_sums}
        for rs in right_sums:
            if target - rs in left_sums:
                return True
        return False
    else:
        # Fallback: use a set-based DP with early exit (n is small in practice here)
        reachable = {0}
        for v in vals:
            new = set()
            for s in reachable:
                ns = s + v
                if ns <= target:
                    new.add(ns)
            reachable |= new
            if target in reachable:
                return True
        return target in reachable

def _check_reciprocal_subsum(target_x, others):
    """Check if 1/target_x equals sum of 1/y for some nonempty subset of others."""
    if not others:
        return False
    all_vals = [target_x] + list(others)
    lcm_val = _lcm_list(all_vals)
    target_int = lcm_val // target_x
    candidate_vals = [lcm_val // y for y in others]
    return _subset_sum_exists(target_int, candidate_vals)

def _is_valid_set(elements):
    """Check that no reciprocal of a selected element equals sum of reciprocals of a subset of the OTHER selected elements."""
    elems = list(elements)
    for i, x in enumerate(elems):
        others = elems[:i] + elems[i+1:]
        if _check_reciprocal_subsum(x, others):
            return False
    return True

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    best_set = list(range(13, 25))  # seed fallback, size 12
    best_size = len(best_set)
    
    # Verify seed is valid (it should be, but just in case)
    if not _is_valid_set(best_set):
        # Find a valid fallback by removing elements
        for x in range(13, 25):
            trial = [y for y in best_set if y != x]
            if _is_valid_set(trial):
                best_set = trial
                best_size = len(trial)
                break
    
    # Branch and bound: try to include larger elements first
    # Order: descending (24, 23, ..., 1)
    candidates = list(range(24, 0, -1))
    
    # State: current chosen set, index into candidates
    chosen = []
    
    def _check_validity_incremental(chosen_list):
        """Check if chosen_list is valid (full check, but chosen is small)."""
        return _is_valid_set(chosen_list)
    
    def _bb(idx, chosen_list, current_size):
        nonlocal best_set, best_size
        if time.time() > deadline:
            return
        # Prune: even if we include all remaining, can't beat best
        remaining = len(candidates) - idx
        if current_size + remaining <= best_size:
            return
        if idx == len(candidates):
            if current_size > best_size:
                best_set = list(chosen_list)
                best_size = current_size
            return
        
        x = candidates[idx]
        
        # Option 1: include x (check validity)
        new_chosen = chosen_list + [x]
        if _check_validity_incremental(new_chosen):
            _bb(idx + 1, new_chosen, current_size + 1)
        
        # Option 2: exclude x
        _bb(idx + 1, chosen_list, current_size)
    
    # Start branch and bound
    _bb(0, [], 0)
    
    # Ensure we return a valid nonempty set
    if not best_set or not _is_valid_set(best_set):
        best_set = [24]
    
    return best_set
# EVOLVE-BLOCK-END
