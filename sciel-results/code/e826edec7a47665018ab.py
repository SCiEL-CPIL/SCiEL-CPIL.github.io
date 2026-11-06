"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
from itertools import combinations

# EVOLVE-BLOCK-START
def _is_subsum_free(S):
    """Check if set S is reciprocal-subsum-free.
    For each x in S, 1/x must not equal sum of 1/y for any non-empty subset of S\{x}."""
    S_list = sorted(S)
    n = len(S_list)
    for i, x in enumerate(S_list):
        target = Fraction(1, x)
        others = S_list[:i] + S_list[i+1:]
        m = len(others)
        # Check all non-empty subsets of others
        for r in range(1, m + 1):
            for subset in combinations(others, r):
                s = sum(Fraction(1, y) for y in subset)
                if s == target:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Try to find the largest valid subset of [1,24]
    # Start with all elements and try to find maximum valid subset
    
    # First, try the full set
    full_set = list(range(1, 25))
    if _is_subsum_free(full_set):
        return full_set
    
    # Try subsets of size 23, 22, ...
    # But this is expensive. Let's use a smarter approach.
    
    # Greedy approach: start with all, remove elements that cause conflicts
    # Or: build up from large to small
    
    # Let's try: start with [24, 23, ..., 1] and add elements one by one
    # Actually, let's try removing the "most problematic" elements first
    
    # Key insight: small numbers are more likely to cause issues
    # Let's try different strategies
    
    # Strategy 1: Try all subsets of size 12+ using pruning
    # Strategy 2: Greedy construction
    
    # Let's first check what sizes are possible
    # Check size 12 (current best)
    # Check if we can get 13, 14, etc.
    
    # Try a greedy approach: start with largest numbers, add smaller ones if valid
    for start_size in range(12, 25):
        if time.time() > deadline:
            break
        # Try to find a valid set of size start_size
        # Use backtracking with pruning
        found = _find_valid_set(start_size, deadline)
        if found is not None:
            return found
    
    # Fallback: return the known good set
    return list(range(13, 25))

def _find_valid_set(target_size, deadline):
    """Try to find a valid subset of [1,24] of size target_size."""
    start_time = time.time()
    
    # Use backtracking: build set element by element
    # Order: try larger elements first (they're less likely to cause conflicts)
    candidates = list(range(24, 0, -1))
    
    best = [None]
    
    def backtrack(current_set, remaining_candidates):
        if time.time() > deadline:
            return
        if len(current_set) == target_size:
            if _is_subsum_free(current_set):
                best[0] = sorted(current_set)
                return
            return
        if not remaining_candidates:
            return
        
        # Pruning: if we can't reach target_size, return
        if len(current_set) + len(remaining_candidates) < target_size:
            return
        
        for i, elem in enumerate(remaining_candidates):
            new_set = current_set + [elem]
            # Quick check: is this partial set still potentially valid?
            # We can check if any element in new_set already violates with others in new_set
            if _partial_check(new_set):
                backtrack(new_set, remaining_candidates[i+1:])
                if best[0] is not None:
                    return
    
    backtrack([], candidates)
    return best[0]

def _partial_check(S):
    """Quick check: for each x in S, verify 1/x is not formed by subset of S\{x}.
    This is the same as full check but on a smaller set."""
    S_list = sorted(S)
    n = len(S_list)
    for i, x in enumerate(S_list):
        target = Fraction(1, x)
        others = S_list[:i] + S_list[i+1:]
        m = len(others)
        for r in range(1, m + 1):
            for subset in combinations(others, r):
                s = sum(Fraction(1, y) for y in subset)
                if s == target:
                    return False
    return True
# EVOLVE-BLOCK-END
