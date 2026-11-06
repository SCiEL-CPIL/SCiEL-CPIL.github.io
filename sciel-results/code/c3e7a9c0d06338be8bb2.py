"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # LCM of 1..22 for exact integer representation
    L = reduce(lcm, range(1, 23))
    
    # Precompute reciprocal as integer: 1/i * L
    recip = [L // i for i in range(1, 23)]
    
    # Best known fallback: 9 elements
    best_result = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    def incremental_check(sums_set, new_idx):
        """
        Incrementally check if adding new_idx keeps all subset sums distinct.
        Returns new sums set if valid, None if collision.
        """
        r = recip[new_idx - 1]
        new_sums = {s + r for s in sums_set}
        if new_sums & sums_set:
            return None
        return sums_set | new_sums
    
    def backtrack(order, pos, current, sums_set):
        """Backtracking search for maximum dissociated subset."""
        nonlocal best_result
        if time.time() > deadline:
            return
        
        # Prune: if we can't beat current best
        remaining = len(order) - pos
        if len(current) + remaining <= len(best_result):
            return
        
        if pos >= len(order):
            if len(current) > len(best_result):
                best_result = current[:]
            return
        
        idx = order[pos]
        
        # Try including idx first (to find large sets faster)
        new_sums = incremental_check(sums_set, idx)
        if new_sums is not None:
            new_current = current + [idx]
            if len(new_current) > len(best_result):
                best_result = new_current[:]
            backtrack(order, pos + 1, new_current, new_sums)
        
        # Try not including idx
        backtrack(order, pos + 1, current, sums_set)
    
    # Use descending order: larger indices first (smaller reciprocals, fewer collisions)
    order = list(range(22, 0, -1))
    backtrack(order, 0, [], {0})
    
    return best_result
# EVOLVE-BLOCK-END
