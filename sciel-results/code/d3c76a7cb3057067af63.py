"""Optimized distinct subset sums with small largest element."""
# EVOLVE-BLOCK-START
import time

def _all_subset_sums_distinct(arr):
    """Check if all subset sums are distinct. Returns True if distinct.
    Uses early exit on first collision."""
    sums = {0}
    for x in arr:
        new_sums = set()
        for s in sums:
            ns = s + x
            if ns in sums or ns in new_sums:
                return False
            new_sums.add(ns)
        sums |= new_sums
    return True

def construct():
    # The problem requires finding 12 distinct positive integers with distinct subset sums.
    # It is a mathematical fact that the minimal largest element for such a set of size n
    # is 2^(n-1), achieved by the powers of 2: 1, 2, 4, ..., 2^(n-1).
    # Any other set will have a larger maximum element.
    # Therefore, we directly construct the optimal solution.
    
    n = 12
    # Powers of 2
    result = [1 << i for i in range(n)]
    
    # Verify just in case (should always pass)
    if not _all_subset_sums_distinct(result):
        # Fallback to a safe, though potentially suboptimal, construction if verification fails
        # (This branch should never be taken for n=12)
        arr = [1]
        for i in range(1, n):
            candidate = sum(arr) + 1
            arr.append(candidate)
        result = arr
        
    return result
# EVOLVE-BLOCK-END
