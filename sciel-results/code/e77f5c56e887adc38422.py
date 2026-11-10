from fractions import Fraction
from itertools import combinations
import time

def _check_dissociated(elements):
    """Check if all subset sums of reciprocals are distinct."""
    n = len(elements)
    if n > 20:
        return False  # Too many to check efficiently
    sums = [Fraction(0)]
    for elem in elements:
        r = Fraction(1, elem)
        new_sums = [s + r for s in sums]
        sums = sums + new_sums
    return len(set(sums)) == len(sums)

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin
    
    # Try to find largest dissociated set
    # Start with all 22, try to remove minimal elements
    # Or build up greedily
    
    # Greedy approach: try to include elements in order of "difficulty"
    # Elements with smaller i have larger reciprocals, potentially more collisions
    
    # Try different orderings and greedy constructions
    best = [1, 2, 3, 5, 7, 11, 13, 17, 19]  # Fallback
    
    # Approach: backtracking to find max size
    # Try sizes from 14 down to 10
    for target_size in range(14, 9, -1):
        if time.time() > deadline:
            break
        # Try to find a set of this size
        # Use greedy with backtracking
        found = _find_set_of_size(target_size, deadline)
        if found:
            best = found
            break
    
    return best

def _find_set_of_size(target_size, deadline):
    """Try to find a dissociated set of given size using backtracking."""
    # Order elements by some heuristic
    # Try including larger reciprocals first (smaller i) as they're more constraining
    candidates = list(range(1, 23))
    
    # Use recursive backtracking
    def backtrack(remaining, current, depth):
        if time.time() > deadline:
            return None
        if len(current) == target_size:
            if _check_dissociated(current):
                return list(current)
            return None
        if not remaining:
            return None
        
        # Pruning: not enough elements left
        if len(remaining) < target_size - len(current):
            return None
        
        for i, elem in enumerate(remaining):
            if time.time() > deadline:
                return None
            new_current = current + [elem]
            # Quick check: if adding this element causes immediate collision
            if _quick_check(new_current):
                result = backtrack(remaining[i+1:], new_current, depth + 1)
                if result:
                    return result
        return None
    
    return backtrack(candidates, [], 0)

def _quick_check(elements):
    """Quick check if subset sums are still distinct for small sets."""
    n = len(elements)
    if n <= 1:
        return True
    if n > 12:
        return True  # Skip expensive check for large sets
    sums = [Fraction(0)]
    for elem in elements:
        r = Fraction(1, elem)
        new_sums = [s + r for s in sums]
        sums = sums + new_sums
    return len(set(sums)) == len(sums)
