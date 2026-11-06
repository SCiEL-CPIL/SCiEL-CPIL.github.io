"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import lcm
from functools import reduce
from itertools import combinations

def _compute_lcm_1_22():
    return reduce(lcm, range(1, 23))

_L = _compute_lcm_1_22()

def _is_dissociated(subset):
    """Check if all subset sums of reciprocals are distinct.
    Multiply by LCM to get integers, then check all subset sums distinct."""
    k = len(subset)
    if k == 0:
        return True
    # Convert to integer values: L // a for each a in subset
    vals = [_L // a for a in subset]
    # Generate all subset sums
    sums = set()
    # Use iterative method
    sums.add(0)
    for v in vals:
        new_sums = set()
        for s in sums:
            ns = s + v
            if ns in sums or ns in new_sums:
                return False
            new_sums.add(ns)
        sums.update(new_sums)
    return True

def _search_max_cardinality():
    """Find the largest subset of [1,22] with all distinct reciprocal subset sums."""
    candidates = list(range(1, 23))
    
    # Try from largest cardinality down
    for k in range(22, 0, -1):
        # Try to find a valid subset of size k
        # Use backtracking with pruning
        found = _backtrack(candidates, k, [])
        if found is not None:
            return found
    return [1]  # fallback

def _backtrack(remaining, target_size, current):
    """Backtrack to find a valid subset of target_size from remaining."""
    if target_size == 0:
        return current
    if len(remaining) < target_size:
        return None
    
    # Try adding each candidate from remaining
    for i, cand in enumerate(remaining):
        new_current = current + [cand]
        # Quick check: is new_current dissociated?
        if _is_dissociated(new_current):
            result = _backtrack(remaining[i+1:], target_size - 1, new_current)
            if result is not None:
                return result
    return None

def construct():
    """Return a nonempty subset of [1,22] with all distinct reciprocal subset sums, maximizing cardinality."""
    result = _search_max_cardinality()
    return result
# EVOLVE-BLOCK-END
