"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

def _subset_sums_distinct_fast(arr):
    """Faster check using incremental sums."""
    n = len(arr)
    sums = [0]
    for x in arr:
        new_sums = [s + x for s in sums]
        all_sums = sums + new_sums
        if len(set(all_sums)) != len(all_sums):
            return False
        sums = all_sums
    return True

def _check_valid_incremental(current_sums, new_elem):
    """Check if adding new_elem to current subset sums preserves distinctness."""
    new_sums = [s + new_elem for s in current_sums]
    all_sums = current_sums + new_sums
    return len(set(all_sums)) == len(all_sums)

def _validate(arr):
    """Validate: 12 distinct positive ints <= max, all subset sums distinct."""
    if len(arr) != 12:
        return False
    if len(set(arr)) != 12:
        return False
    if any(x <= 0 for x in arr):
        return False
    if any(x > 1048576 for x in arr):
        return False
    return _subset_sums_distinct_fast(arr)

def _score(arr):
    """Score is the largest element (lower is better)."""
    return max(arr)

def _try_reduce_max(arr, deadline):
    """Try to reduce elements in arr while maintaining validity."""
    arr = sorted(arr)
    n = len(arr)
    
    # Try to reduce each element from largest to smallest
    for target_idx in range(n - 1, -1, -1):
        current_val = arr[target_idx]
        # Try smaller values
        for new_val in range(current_val - 1, 0, -1):
            if time.time() > deadline:
                return arr
            if new_val in arr:
                continue
            test_arr = arr.copy()
            test_arr[target_idx] = new_val
            test_arr.sort()
            if _validate(test_arr):
                arr = test_arr
                break
        if time.time() > deadline:
            break
    
    return arr

def _greedy_construct(start_val=1, deadline=None):
    """Greedy construction: pick smallest valid element at each step."""
    if deadline is None:
        deadline = time.time() + 5
    
    n = 12
    elements = [start_val]
    subset_sums = [0, start_val]
    
    for step in range(1, n):
        if time.time() > deadline:
            return None
        
        # Find smallest positive integer not in elements that preserves distinctness
        elem_set = set(elements)
        candidate = 1
        while True:
            if time.time() > deadline:
                return None
            if candidate not in elem_set:
                if _check_valid_incremental(subset_sums, candidate):
                    elements.append(candidate)
                    subset_sums = subset_sums + [s + candidate for s in subset_sums]
                    break
            candidate += 1
            if candidate > 1048576:
                return None
    
    return elements

def construct():
    deadline = time.time() + 85  # 85s out of 90s budget, 5s safety margin
    
    n = 12
    # Power of 2 solution as baseline
    best = [1 << i for i in range(n)]
    best_val = max(best)  # 2048
    
    # Try greedy construction with different starting values
    start_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50]
    
    for start_val in start_vals:
        if time.time() > deadline - 10:
            break
        
        result = _greedy_construct(start_val, deadline - 5)
        if result is not None and _validate(result):
            val = max(result)
            if val < best_val:
                best_val = val
                best = result
    
    # Local improvement pass: try to reduce elements in the best solution
    if time.time() < deadline - 5:
        improved = _try_reduce_max(best, deadline - 3)
        if _validate(improved) and max(improved) < best_val:
            best = improved
            best_val = max(best)
    
    # Final validation
    if _validate(best):
        return best
    
    # Fallback: return power of 2 solution
    return [1 << i for i in range(n)]
