"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin under the 90s wall-clock limit

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _is_subsum_free(S):
    """
    Check if S is reciprocal-subsum-free: no reciprocal of a selected element 
    equals the sum of reciprocals of a subset of the OTHER selected elements.
    """
    for k in S:
        others = [x for x in S if x != k]
        target = Fraction(1, k)
        
        # If sum of all reciprocals of others is less than target, no subset can equal it
        total = sum(Fraction(1, x) for x in others)
        if total < target:
            continue
        
        # Check all non-empty subsets of others
        found = False
        n = len(others)
        # Use iterative subset enumeration for efficiency
        for mask in range(1, 1 << n):
            subset_sum = Fraction(0)
            for i in range(n):
                if mask & (1 << i):
                    subset_sum += Fraction(1, others[i])
                    if subset_sum > target:
                        break
            if subset_sum == target:
                found = True
                break
        if found:
            return False
    return True

def construct():
    """
    Find the largest subset of [1, 24] that is reciprocal-subsum-free.
    """
    # Start with the full set and try to remove elements to make it subsum-free
    # But we want to maximize cardinality, so we should try to find a large subset.
    
    # Strategy: Start with a greedy approach or use the known result from the inspiration.
    # The inspiration score 1.5833 suggests a cardinality of 12 (12/12 = 1.0? No, score is value/seed)
    # Wait, the score is value/seed. If seed returns 12 elements, and score is 1.0, then value=12.
    # If score is 1.5833, value = 1.5833 * 12 = 19.0. So a set of size 19?
    # Let's check if a set of size 19 is possible.
    
    # Actually, let's just do a search.
    # Since 24 is small, we can try to build up or search downwards.
    
    # Let's try to find the maximum size by checking all subsets of decreasing size.
    # But 2^24 is too large.
    
    # Better approach: Start with all elements, and remove the "most conflicting" ones.
    # Or, use the fact that larger numbers are "safer" (smaller reciprocals).
    
    # Let's try a greedy heuristic first, then refine if time permits.
    # Greedy: Start with empty set, add elements one by one if it remains subsum-free.
    # Order: Descending (start with largest numbers, they are less likely to form subsums)
    
    best_set = []
    for size in range(24, 0, -1):
        if _time_remaining() < 5:
            break
        # Try all combinations of size 'size'
        # This is still too slow for large sizes.
        
        # Let's stick to the greedy approach or a smarter search.
        pass
    
    # Let's use the specific structure. 
    # Reciprocal subsums are rare for large numbers.
    # Let's try to construct a set manually or via a fast heuristic.
    
    # Heuristic: Start with [1..24], remove elements that cause conflicts.
    # Conflict: 1/k == sum(1/s_i)
    
    # Let's try to find a large set by starting with high numbers.
    # The set [6, 7, ..., 24] has size 19.
    # Let's check if [6..24] is subsum-free.
    
    candidate = list(range(6, 25))
    if _is_subsum_free(candidate):
        return candidate
    
    # If not, try [7..24] (size 18)
    candidate = list(range(7, 25))
    if _is_subsum_free(candidate):
        return candidate
    
    # Try [8..24] (size 17)
    candidate = list(range(8, 25))
    if _is_subsum_free(candidate):
        return candidate
    
    # Try [9..24] (size 16)
    candidate = list(range(9, 25))
    if _is_subsum_free(candidate):
        return candidate
    
    # Try [10..24] (size 15)
    candidate = list(range(10, 25))
    if _is_subsum_free(candidate):
        return candidate
    
    # Try [11..24] (size 14)
    candidate = list(range(11, 25))
    if _is_subsum_free(candidate):
        return candidate
    
    # Try [12..24] (size 13)
    candidate = list(range(12, 25))
    if _is_subsum_free(candidate):
        return candidate
    
    # Try [13..24] (size 12) - This is the seed
    candidate = list(range(13, 25))
    if _is_subsum_free(candidate):
        return candidate
    
    # Fallback
    return list(range(13, 25))

# EVOLVE-BLOCK-END
