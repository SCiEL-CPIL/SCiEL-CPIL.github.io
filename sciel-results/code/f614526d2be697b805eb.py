"""Large reciprocal-subsum-free sets: maximize cardinality subset of [1,24]."""
import time
from itertools import combinations

_DEADLINE = 90.0
_MARGIN = 5.0  # seconds to spare

def _recip_sum_free_check(elements):
    """
    Check if a set of elements is reciprocal-subsum-free.
    For each element x in the set, 1/x must NOT equal sum of 1/y for any
    non-empty subset of the OTHER elements.
    """
    n = len(elements)
    if n == 0:
        return True
    
    # Precompute reciprocals
    recips = [1.0 / x for x in elements]
    
    # For each target index i, check all subsets of other indices
    for i in range(n):
        target = recips[i]
        # Get indices of all other elements
        others = [j for j in range(n) if j != i]
        m = len(others)
        # Check all non-empty subsets of others
        # For efficiency, we can use meet-in-the-middle for larger sets
        if m <= 20:
            # Direct subset enumeration
            for mask in range(1, 1 << m):
                s = 0.0
                for bit in range(m):
                    if mask & (1 << bit):
                        s += recips[others[bit]]
                if abs(s - target) < 1e-12:
                    return False
        else:
            # Meet in the middle
            half = m // 2
            left_indices = others[:half]
            right_indices = others[half:]
            
            left_sums = [0.0]
            for idx in left_indices:
                left_sums += [s + recips[idx] for s in left_sums]
            
            right_sums = [0.0]
            for idx in right_indices:
                right_sums += [s + recips[idx] for s in right_sums]
            
            # Sort right_sums for binary search
            right_sums_sorted = sorted(right_sums)
            
            for ls in left_sums:
                needed = target - ls
                # Binary search for needed in right_sums_sorted
                lo, hi = 0, len(right_sums_sorted)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if right_sums_sorted[mid] < needed:
                        lo = mid + 1
                    else:
                        hi = mid
                if lo < len(right_sums_sorted) and abs(right_sums_sorted[lo] - needed) < 1e-12:
                    # We need to ensure the subset is non-empty
                    # If ls == 0 and needed == 0, that means empty subset on both sides
                    # But we excluded empty subset by starting from mask=1 in direct case
                    # In meet-in-middle, we need to check: if ls==0 and right sum==0, both empty -> invalid
                    if ls == 0.0 and needed == 0.0:
                        continue
                    return False
    
    return True

def _is_valid(elements):
    """Check if the set is valid (reciprocal-subsum-free)."""
    if not elements:
        return False
    return _recip_sum_free_check(elements)

def _greedy_largest_first():
    """
    Greedy: start with all elements 24 down to 1, add one by one.
    Start from largest (safest targets) and try to add each.
    """
    selected = []
    for x in range(24, 0, -1):
        candidate = selected + [x]
        if _is_valid(candidate):
            selected.append(x)
    return selected

def _local_search_improve(initial):
    """
    Try to improve the set by:
    1. Adding elements that were skipped
    2. Swapping elements to allow more additions
    """
    selected = set(initial)
    all_elements = set(range(1, 25))
    not_selected = all_elements - selected
    
    improved = True
    while improved:
        improved = False
        # Try adding each unselected element
        for x in sorted(not_selected, reverse=True):
            candidate = selected | {x}
            if _is_valid(list(candidate)):
                selected = candidate
                not_selected = all_elements - selected
                improved = True
                break
        if improved:
            continue
        
        # Try swaps: remove one selected, add one unselected
        for x in list(selected):
            for y in sorted(not_selected, reverse=True):
                candidate = (selected - {x}) | {y}
                if len(candidate) > len(selected):
                    continue  # No benefit
                if _is_valid(list(candidate)):
                    # Check if this swap helps us add more later
                    selected = candidate
                    not_selected = all_elements - selected
                    improved = True
                    break
            if improved:
                break
    
    return list(selected)

def _randomized_search():
    """
    Try multiple random orderings and greedy constructions,
    then local search.
    """
    import random
    rng = random.Random(42)
    
    best = []
    start_time = time.time()
    
    # First, deterministic greedy
    det = _greedy_largest_first()
    det = _local_search_improve(det)
    best = det
    
    # Try different random seeds and orderings
    for trial in range(200):
        if time.time() - start_time > _DEADLINE - _MARGIN:
            break
        
        # Random permutation of elements
        elems = list(range(1, 25))
        rng.shuffle(elems)
        
        # Greedy in this order
        selected = []
        for x in elems:
            candidate = selected + [x]
            if _is_valid(candidate):
                selected.append(x)
        
        # Local search
        selected = _local_search_improve(selected)
        
        if len(selected) > len(best):
            best = selected
            # Try to improve further
            best = _local_search_improve(best)
    
    return best

def _systematic_search():
    """
    More systematic approach: try to build from largest elements.
    Large elements have small reciprocals, making them harder to subsume.
    """
    start_time = time.time()
    best = []
    
    # Strategy: start with top-k largest, then try to add smaller ones
    for top_k in range(12, 5, -1):
        if time.time() - start_time > _DEADLINE - _MARGIN:
            break
        
        base = list(range(25 - top_k, 25))  # largest top_k elements
        if not _is_valid(base):
            continue
        
        # Try to add remaining elements
        remaining = list(range(1, 25 - top_k))
        selected = base[:]
        
        for x in remaining:
            if time.time() - start_time > _DEADLINE - _MARGIN:
                break
            candidate = selected + [x]
            if _is_valid(candidate):
                selected.append(x)
        
        if len(selected) > len(best):
            best = selected
            best = _local_search_improve(best)
    
    return best

def construct():
    start_time = time.time()
    
    # Quick deadline check
    if time.time() - start_time > _DEADLINE - _MARGIN:
        return list(range(13, 25))
    
    # Strategy 1: Greedy largest-first
    result1 = _greedy_largest_first()
    result1 = _local_search_improve(result1)
    
    if time.time() - start_time > _DEADLINE - _MARGIN:
        return result1
    
    # Strategy 2: Systematic search
    result2 = _systematic_search()
    
    if time.time() - start_time > _DEADLINE - _MARGIN:
        return max(result1, result2, key=len)
    
    # Strategy 3: Randomized search
    result3 = _randomized_search()
    
    # Return the best
    best = max([result1, result2, result3], key=len)
    
    # Final validation
    if not _is_valid(best):
        # Fallback to a known valid set
        return list(range(13, 25))
    
    return best
