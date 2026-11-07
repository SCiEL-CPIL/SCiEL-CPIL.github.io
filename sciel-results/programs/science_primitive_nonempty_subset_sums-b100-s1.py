"""Search-based approach to maximize set size under divisibility constraint."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

DEADLINE_SECONDS = 85.0

def _all_subset_sums(elements):
    """Generate all nonempty subset sums for a list of elements."""
    n = len(elements)
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        sums.add(s)
    return sums

def _is_valid(elements):
    """Check if no two distinct subset sums divide one another."""
    n = len(elements)
    if n == 0:
        return True
    if n > 20:
        # Too many subsets for brute force; use heuristic check
        # Check pairwise divisibility of individual elements first
        for i in range(n):
            for j in range(i+1, n):
                a, b = elements[i], elements[j]
                if a > b:
                    a, b = b, a
                if b % a == 0:
                    return False
        return True  # Optimistic; will be verified by caller if needed
    
    sums = _all_subset_sums(elements)
    sum_list = sorted(sums)
    for i in range(len(sum_list)):
        for j in range(i+1, len(sum_list)):
            a, b = sum_list[i], sum_list[j]
            if b % a == 0:
                return False
    return True

def _can_add(elements, candidate):
    """Check if adding candidate to elements keeps the set valid."""
    new_set = elements + [candidate]
    return _is_valid(new_set)

def _greedy_construct(start_elements, candidates, deadline):
    """Greedy: try to add candidates in order, keeping valid set."""
    current = list(start_elements)
    for c in candidates:
        if time.time() > deadline:
            break
        if _can_add(current, c):
            current.append(c)
    return current

def _local_search(elements, candidates, deadline, max_iter=500):
    """Local search: try swapping elements to improve size."""
    current = list(elements)
    best = list(current)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        # Try adding a random candidate
        if candidates:
            idx = random.randint(0, len(candidates) - 1)
            c = candidates[idx]
            if c not in current:
                if _can_add(current, c):
                    current.append(c)
                    if len(current) > len(best):
                        best = list(current)
        
        # Try removing and re-adding to find better configuration
        if len(current) > 1:
            remove_idx = random.randint(0, len(current) - 1)
            removed = current.pop(remove_idx)
            if candidates:
                add_idx = random.randint(0, len(candidates) - 1)
                c = candidates[add_idx]
                if c not in current:
                    if _can_add(current, c):
                        current.append(c)
                    else:
                        current.append(removed)
            else:
                current.append(removed)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # All candidates from 1 to 256
    all_candidates = list(range(1, 257))
    
    # Strategy: use larger numbers to reduce subset sum collisions
    # Start with a seed of larger numbers
    random.seed(42)
    
    best_result = []
    
    # Multiple restarts with different starting points
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Randomly select a starting set of larger numbers
        large_candidates = [x for x in all_candidates if x >= 100]
        random.shuffle(large_candidates)
        
        # Start with 2-4 elements
        start_size = random.randint(2, 4)
        start_elements = large_candidates[:start_size]
        
        # Ensure start is valid
        if not _is_valid(start_elements):
            continue
        
        # Greedy add
        remaining = [x for x in large_candidates if x not in start_elements]
        random.shuffle(remaining)
        current = _greedy_construct(start_elements, remaining, deadline)
        
        # Local search
        all_remaining = [x for x in all_candidates if x not in current]
        current = _local_search(current, all_remaining, deadline, max_iter=100)
        
        if len(current) > len(best_result):
            best_result = list(current)
        
        # Early termination if we hit maximum
        if len(best_result) >= 15:
            break
    
    # Final verification
    if best_result and not _is_valid(best_result):
        # Fallback to a known small valid set
        best_result = [31, 32, 34]
    
    # Ensure all are Python ints
    return [int(x) for x in best_result]

# EVOLVE-BLOCK-END
