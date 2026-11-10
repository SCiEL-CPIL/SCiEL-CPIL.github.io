"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

_start_time = time.time()
_DEADLINE = 85.0  # safety margin before 90s kill

def _check_dissociated(subset):
    """Check if all 2^n reciprocal subset sums are distinct using exact rational arithmetic."""
    n = len(subset)
    # Compute all 2^n subset sums of {1/s for s in subset}
    reciprocals = [Fraction(1, s) for s in subset]
    
    # Use iterative method to generate all subset sums
    sums = {Fraction(0, 1)}
    for r in reciprocals:
        new_sums = set()
        for s in sums:
            new_sums.add(s + r)
        sums |= new_sums
        if len(sums) != 2 ** (len(sums) and (list(sums).index(Fraction(0,1)) + 1) if False else 0):
            pass
        # After adding k-th element, we should have 2^k distinct sums
        k = subset.index(s) + 1 if s in subset else 0
        # Actually, let's track count properly
        # After processing element i (0-indexed), we should have 2^(i+1) distinct sums
        # But we're iterating, so let's just check at the end
        if len(sums) < 2 ** (reciprocals.index(r) + 1):
            return False
    
    return len(sums) == 2 ** n

def _check_dissociated_fast(subset):
    """Faster check: build subset sums incrementally, check cardinality at each step."""
    n = len(subset)
    reciprocals = [Fraction(1, s) for s in subset]
    
    sums = {Fraction(0, 1)}
    for i, r in enumerate(reciprocals):
        new_sums = set()
        for s in sums:
            new_sums.add(s + r)
        sums |= new_sums
        expected = 2 ** (i + 1)
        if len(sums) != expected:
            return False
    return True

def _search_max_dissociated(deadline):
    """Search for maximum cardinality dissociated subset of [1,22]."""
    universe = list(range(1, 23))
    
    # Try from largest size down
    for size in range(22, 0, -1):
        if time.time() > deadline:
            break
        
        found = False
        for combo in combinations(universe, size):
            if time.time() > deadline:
                return None
            
            if _check_dissociated_fast(combo):
                return list(combo)
        
        # If we exhausted all combos of this size without finding one, continue to smaller
        # But this could be very slow for large sizes
    
    return None

def construct():
    global _start_time
    current_time = time.time()
    remaining = _DEADLINE - current_time
    
    if remaining <= 0:
        # Fallback to known good set
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    # Known good sets to try as starting points for extension
    known_sets = [
        [1, 2, 3, 5, 7, 11, 13, 17, 19],  # 9 elements
    ]
    
    # First, try to find a larger set via systematic search
    # For efficiency, try sizes 12, 11, 10 first (more likely to exist and findable)
    
    # Actually, let's be smarter: try to extend known sets
    # Or just do a direct search for high sizes
    
    best_found = None
    
    # Try sizes from high to low, but with time limits per size
    for size in [14, 13, 12, 11, 10]:
        if time.time() > _DEADLINE - 5:  # Keep 5s safety margin
            break
        
        size_deadline = min(_DEADLINE - 5, time.time() + 15)  # 15s per size attempt
        
        found = _search_for_size(size, list(range(1, 23)), size_deadline)
        if found is not None:
            best_found = found
            break
    
    if best_found is not None:
        return best_found
    
    # Fallback: verify and return known set
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]

def _search_for_size(target_size, universe, deadline):
    """Search for a dissociated subset of given size."""
    from itertools import combinations
    
    for combo in combinations(universe, target_size):
        if time.time() > deadline:
            return None
        if _check_dissociated_fast(combo):
            return list(combo)
    return None

# EVOLVE-BLOCK-END
