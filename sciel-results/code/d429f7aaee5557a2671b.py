"""Find maximum cardinality dissociated reciprocal subset of [1,22]."""
import time
from fractions import Fraction
from itertools import combinations

# EVOLVE-BLOCK-START
def _is_dissociated(elements):
    """Check if all subset sums of reciprocals are distinct."""
    n = len(elements)
    recips = [Fraction(1, e) for e in elements]
    sums = set()
    # 2^n subset sums
    for mask in range(1 << n):
        s = Fraction(0, 1)
        for i in range(n):
            if mask & (1 << i):
                s += recips[i]
        if s in sums:
            return False
        sums.add(s)
    return True

def _is_dissociated_fast(elements):
    """Faster check using incremental subset sum generation."""
    n = len(elements)
    if n == 0:
        return True
    recips = [Fraction(1, e) for e in elements]
    # Generate all subset sums incrementally
    sums = [Fraction(0, 1)]
    for r in recips:
        new_sums = [s + r for s in sums]
        # Check for collisions
        all_sums = sums + new_sums
        if len(set(all_sums)) != len(all_sums):
            return False
        sums = all_sums
    return True

def _try_add(element, current_elements):
    """Check if adding element to current_elements keeps dissociation."""
    return _is_dissociated_fast(current_elements + [element])

def _backtrack_search(elements_pool, current_set, best_set, deadline, max_size_target):
    """Backtracking search to find largest dissociated set."""
    if time.time() > deadline:
        return best_set
    if len(current_set) >= len(best_set):
        # Only update if strictly better or equal size but we want max
        if len(current_set) > len(best_set):
            best_set = list(current_set)
    # If we've reached target size, no need to continue
    if len(current_set) >= max_size_target:
        return best_set
    
    for i, elem in enumerate(elements_pool):
        if elem in current_set:
            continue
        # Try adding elem
        new_set = current_set + [elem]
        if _is_dissociated_fast(new_set):
            best_set = _backtrack_search(elements_pool, new_set, best_set, deadline, max_size_target)
            if time.time() > deadline:
                return best_set
            # If we found a set of target size, we can stop
            if len(best_set) >= max_size_target:
                return best_set
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline for 90s limit with safety margin
    
    # Known valid 9-element set from seed
    seed_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best = list(seed_set)
    
    # Try to extend to size 10, 11, 12...
    # Strategy: greedy + backtracking
    # First try simple greedy: start with seed, try adding each remaining element
    remaining = [i for i in range(1, 23) if i not in seed_set]
    
    # Try adding elements one by one (greedy)
    current = list(seed_set)
    for elem in remaining:
        if time.time() > deadline:
            break
        if _is_dissociated_fast(current + [elem]):
            current.append(elem)
            if len(current) > len(best):
                best = list(current)
    
    # Now try to improve further with backtracking
    # Try different starting points and search strategies
    
    # Strategy 2: Backtracking from scratch, preferring larger elements
    # Larger elements have smaller reciprocals, making collisions less likely
    all_elements = list(range(1, 23))
    
    # Try to find a set of size 10+
    # Use a smarter approach: try all combinations of size 10, 11, 12...
    # But that's too many. Use backtracking with pruning.
    
    # Better: iterative deepening from current best
    target_size = len(best) + 1
    while target_size <= 15 and time.time() < deadline:
        # Try to find a set of target_size
        # Use backtracking with the constraint that we build up to target_size
        found = _find_set_of_size(target_size, deadline)
        if found is not None:
            best = found
            target_size += 1
        else:
            break
    
    return best

def _find_set_of_size(target_size, deadline):
    """Find a dissociated set of exactly target_size elements from [1,22]."""
    # Use backtracking with ordering that prefers larger elements
    # (smaller reciprocals are less likely to cause collisions)
    
    def backtrack(remaining_pool, current, depth):
        if time.time() > deadline:
            return None
        if len(current) == target_size:
            if _is_dissociated_fast(current):
                return list(current)
            return None
        
        # Pruning: not enough elements left
        if len(remaining_pool) < target_size - len(current):
            return None
        
        # Try each element in remaining_pool
        for i, elem in enumerate(remaining_pool):
            if time.time() > deadline:
                return None
            new_current = current + [elem]
            # Quick check: is this partial set dissociated?
            if _is_dissociated_fast(new_current):
                result = backtrack(remaining_pool[i+1:], new_current, depth + 1)
                if result is not None:
                    return result
        return None
    
    # Start with larger elements first (they have smaller reciprocals)
    pool = list(range(22, 0, -1))  # 22, 21, 20, ..., 1
    return backtrack(pool, [], 0)

# EVOLVE-BLOCK-END
