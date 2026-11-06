"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from functools import reduce

def _compute_lcm_1_22():
    return reduce(lcm, range(1, 23))

_L = _compute_lcm_1_22()

def _is_dissociated(sums, v):
    """Check if adding v keeps subset sums distinct.
    Equivalent to checking if sums and sums+v are disjoint."""
    for s in sums:
        if (s + v) in sums:
            return False
    return True

def _greedy_init(ordering, vals, deadline):
    """Greedy: add elements in order if they keep dissociation."""
    current_sums = {0}
    current_set = []
    for i, a in enumerate(ordering):
        if time.time() > deadline:
            break
        v = vals[i]
        if _is_dissociated(current_sums, v):
            current_sums = current_sums | {s + v for s in current_sums}
            current_set.append(a)
    return current_set, current_sums

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    # Multiple orderings to try
    orderings = [
        [a for a in range(1, 23)],      # a ascending (v descending)
        [a for a in range(22, 0, -1)],  # a descending (v ascending)
    ]
    
    best_subset = []
    best_size = 0
    
    for ordering in orderings:
        if time.time() > deadline:
            break
            
        vals = [_L // a for a in ordering]
        
        # Greedy initialization
        greedy_set, _ = _greedy_init(ordering, vals, deadline)
        if len(greedy_set) > best_size:
            best_size = len(greedy_set)
            best_subset = greedy_set.copy()
        
        # Backtracking search with pruning
        n = len(ordering)
        
        def backtrack(idx, current_sums, current_set):
            nonlocal best_subset, best_size
            
            if time.time() > deadline:
                return
                
            # Pruning: can't beat best
            if len(current_set) + (n - idx) <= best_size:
                return
                
            # Update best if current is better
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_subset = current_set.copy()
                
            if idx == n:
                return
                
            a = ordering[idx]
            v = vals[idx]
            
            # Try including a first (if valid)
            if _is_dissociated(current_sums, v):
                new_sums = current_sums | {s + v for s in current_sums}
                backtrack(idx + 1, new_sums, current_set + [a])
            
            # Try excluding a
            backtrack(idx + 1, current_sums, current_set)
        
        backtrack(0, {0}, [])
        
    if not best_subset:
        return [1, 2, 4, 8, 16]
        
    return sorted(best_subset)
# EVOLVE-BLOCK-END
