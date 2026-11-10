"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time
from itertools import combinations

def _compute_cardinality_sums(elements):
    """Compute set of subset sums for each cardinality."""
    n = len(elements)
    # sums[k] = set of all sums of k-element subsets
    sums = [set() for _ in range(n + 1)]
    sums[0].add(0)
    for i, x in enumerate(elements):
        for k in range(i, 0, -1):
            for s in list(sums[k - 1]):
                sums[k].add(s + x)
    return sums

def _is_cardinality_separated(sums):
    """Check that no sum appears in two different cardinality buckets."""
    seen = {}
    for k, s in enumerate(sums):
        if k == 0:
            continue
        for val in s:
            if val in seen and seen[val] != k:
                return False
            seen[val] = k
    return True

def _check_add_valid(elements, sums, x):
    """Check if adding x to elements preserves cardinality separation."""
    n = len(elements)
    # New sums: old sums plus sums with x
    new_sums = [set() for _ in range(n + 2)]
    for k in range(n + 1):
        new_sums[k] = set(sums[k])
    for k in range(n, -1, -1):
        for s in sums[k]:
            new_sums[k + 1].add(s + x)
    return _is_cardinality_separated(new_sums)

def construct():
    """
    Constructs a set of distinct integers in 1..256 such that subset sums
    with different cardinalities are distinct.
    """
    deadline = time.time() + 85.0  # Safety margin before 90s limit
    
    # Strategy: Greedy with minimum-degree heuristic
    # Try multiple orderings and pick the best result
    
    candidates = list(range(1, 257))
    best_set = list(range(226, 257))  # Fallback: 31 elements
    
    def greedy_construct(order):
        elements = []
        sums = [set() for _ in range(61)]
        sums[0].add(0)
        
        for x in order:
            if time.time() > deadline:
                break
            if _check_add_valid(elements, sums, x):
                elements.append(x)
                n = len(elements)
                # Update sums
                for k in range(n - 1, -1, -1):
                    for s in list(sums[k]):
                        sums[k + 1].add(s + x)
        
        return elements
    
    # Try ascending order
    result = greedy_construct(candidates)
    if len(result) > len(best_set):
        best_set = result
    
    # Try descending order
    result = greedy_construct(candidates[::-1])
    if len(result) > len(best_set):
        best_set = result
    
    # Try minimum-degree heuristic: at each step, pick the candidate
    # that conflicts with the fewest remaining candidates
    # Approximate by trying elements in order of "spread"
    
    # Try a few different heuristics
    # Heuristic: prefer elements that are "spread out"
    import random
    random.seed(42)
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        # Create a shuffled order with bias
        shuffled = candidates[:]
        random.shuffle(shuffled)
        result = greedy_construct(shuffled)
        if len(result) > len(best_set):
            best_set = result
    
    # Try to extend the best set found by adding more elements
    if time.time() < deadline:
        # Try to add elements not in best_set
        current = list(best_set)
        sums = _compute_cardinality_sums(current)
        for x in candidates:
            if x in current:
                continue
            if _check_add_valid(current, sums, x):
                current.append(x)
                sums = _compute_cardinality_sums(current)
        
        if len(current) > len(best_set):
            best_set = current
    
    return best_set

# EVOLVE-BLOCK-END
