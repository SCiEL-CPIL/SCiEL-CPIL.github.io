"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import itertools
from functools import lru_cache

def _subset_sums_distinct(arr):
    """Check if all subset sums are distinct. Returns True if valid."""
    n = len(arr)
    total = sum(arr)
    # Use a set to track seen sums
    seen = set()
    # There are 2^n subsets
    # For n=12, that's 4096, which is manageable
    for mask in range(1 << n):
        s = 0
        # Compute sum for this mask
        for i in range(n):
            if mask & (1 << i):
                s += arr[i]
        if s in seen:
            return False
        seen.add(s)
    return True

def _is_valid(arr):
    """Check if the array satisfies all constraints."""
    n = len(arr)
    if n != 12:
        return False
    if len(set(arr)) != n:
        return False
    if any(x <= 0 for x in arr):
        return False
    if max(arr) > 1048576:
        return False
    return _subset_sums_distinct(arr)

def _greedy_construct():
    """
    Greedy construction: pick smallest possible positive integers one by one
    such that all subset sums remain distinct.
    
    Strategy: Sort candidates in increasing order. For each position, try the
    smallest unused positive integer that maintains the distinct subset sum property.
    """
    n = 12
    arr = []
    used = set()
    
    # We'll try to build the array greedily
    # At each step, try candidates from 1 upward
    # But we need to be smart about it
    
    # Actually, a better approach: use the known optimal construction
    # The optimal for "distinct subset sums" (superincreasing-like but optimized)
    # is related to the problem of finding a set with all subset sums distinct.
    
    # Known result: The minimal maximum element for n elements with distinct subset sums
    # is achieved by certain constructions. For n=12, let's try a greedy approach.
    
    # Greedy: at each step, pick the smallest positive integer not yet used
    # that keeps all subset sums distinct.
    
    for i in range(n):
        candidate = 1
        while True:
            if candidate in used:
                candidate += 1
                continue
            new_arr = arr + [candidate]
            if _subset_sums_distinct(new_arr):
                arr.append(candidate)
                used.add(candidate)
                break
            candidate += 1
    
    return arr

def _local_search(arr):
    """
    Try to improve the array via local search.
    """
    n = len(arr)
    best = arr[:]
    best_max = max(best)
    
    for iteration in range(10000):
        # Try swapping or replacing elements
        improved = False
        for i in range(n):
            # Try to replace arr[i] with a smaller value
            for new_val in range(1, arr[i]):
                if new_val in arr and new_val != arr[i]:
                    continue
                new_arr = arr[:]
                new_arr[i] = new_val
                if _is_valid(new_arr):
                    arr = new_arr
                    best = arr[:]
                    best_max = max(best)
                    improved = True
                    break
        if not improved:
            break
    
    return best

def construct():
    """
    Construct exactly 12 distinct positive Python integers <= 1048576
    such that every subset has a different sum. Minimize the largest integer.
    """
    # Start with greedy construction
    arr = _greedy_construct()
    
    # Try local search to improve
    arr = _local_search(arr)
    
    # Verify
    if not _is_valid(arr):
        # Fallback to power of 2
        return [1 << i for i in range(12)]
    
    return sorted(arr)
# EVOLVE-BLOCK-END
