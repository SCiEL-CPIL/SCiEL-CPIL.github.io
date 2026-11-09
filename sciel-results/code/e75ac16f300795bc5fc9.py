"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from functools import lru_cache

def _all_subset_sums(elements):
    """Return set of all non-empty subset sums."""
    sums = set()
    n = len(elements)
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        sums.add(s)
    return sums

def _is_valid(elements):
    """Check if all non-empty subset sums are pairwise non-dividing."""
    sums = _all_subset_sums(elements)
    sums_list = sorted(sums)
    n = len(sums_list)
    for i in range(n):
        si = sums_list[i]
        for j in range(i + 1, n):
            sj = sums_list[j]
            if sj % si == 0:
                return False
    return True

def _try_add(elements, candidate, deadline):
    """Try to add candidate to elements, check validity."""
    new_elements = elements + [candidate]
    if time.time() > deadline:
        return False
    return _is_valid(new_elements)

def construct():
    deadline = time.time() + 80  # safety margin before 90s limit
    
    # Strategy: greedy build with backtracking
    # Start with small numbers, try to build up a large valid set
    
    best = [31, 32, 34]  # fallback
    
    # Try to build a set greedily
    # Use a heuristic: prefer numbers that are "incompatible" with existing sums
    
    elements = []
    used = set()
    
    # Try candidates in various orders
    # First, try a pure greedy approach
    for candidate in range(1, 257):
        if time.time() > deadline:
            break
        if candidate in used:
            continue
        new_elements = elements + [candidate]
        if _is_valid(new_elements):
            elements = new_elements
            used.add(candidate)
            if len(elements) > len(best):
                best = list(elements)
    
    # If greedy didn't find enough, try randomized restarts
    if len(best) < 10:
        import random
        random.seed(42)
        for trial in range(200):
            if time.time() > deadline:
                break
            elements = []
            used = set()
            candidates = list(range(1, 257))
            random.shuffle(candidates)
            for candidate in candidates:
                if time.time() > deadline:
                    break
                new_elements = elements + [candidate]
                if _is_valid(new_elements):
                    elements = new_elements
                    used.add(candidate)
                    if len(elements) > len(best):
                        best = list(elements)
    
    # Final validation
    if not _is_valid(best):
        best = [31, 32, 34]
    
    return list(best)
# EVOLVE-BLOCK-END
