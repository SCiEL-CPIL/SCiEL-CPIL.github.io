"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from functools import reduce

def _compute_lcm_1_22():
    return reduce(lcm, range(1, 23))

_L = _compute_lcm_1_22()

def _is_dissociated(subset):
    """Check if all subset sums of reciprocals are distinct using LCM integer arithmetic."""
    k = len(subset)
    if k == 0:
        return True
    vals = [_L // a for a in subset]
    sums = {0}
    for v in vals:
        new_sums = set()
        for s in sums:
            ns = s + v
            if ns in sums or ns in new_sums:
                return False
            new_sums.add(ns)
        sums.update(new_sums)
    return True

def _backtrack(max_cardinality, deadline):
    """Backtracking search to find a dissociated subset of maximum cardinality."""
    best = []
    # Try from larger cardinality down
    for target_size in range(max_cardinality, 0, -1):
        if time.time() > deadline:
            break
        # Use backtracking with pruning
        current = []
        used_sums = {0}
        
        def backtrack(start, remaining, target):
            nonlocal best
            if time.time() > deadline:
                return False
            if len(current) == target:
                # Verify the full set is dissociated
                if _is_dissociated(current):
                    best = list(current)
                    return True
                return False
            
            if start >= 22:
                return False
            
            # Pruning: not enough elements left
            if 22 - start < remaining:
                return False
            
            for i in range(start, 22 - remaining + 1):
                a = i + 1  # Elements are 1..22
                v = _L // a
                
                # Check if adding v creates a collision
                collision = False
                for s in used_sums:
                    ns = s + v
                    if ns in used_sums:
                        collision = True
                        break
                
                if not collision:
                    # Add the new sums
                    new_sums = set()
                    for s in used_sums:
                        new_sums.add(s + v)
                    
                    current.append(a)
                    old_sums = used_sums
                    used_sums = used_sums | new_sums
                    
                    if backtrack(i + 1, remaining - 1, target):
                        return True
                    
                    used_sums = old_sums
                    current.pop()
            
            return False
        
        if backtrack(0, target_size, target_size):
            return best
    
    return best

def construct():
    """Find a nonempty subset of [1,22] with all reciprocal subset sums distinct.
    Maximize cardinality."""
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    # Try to find maximum cardinality
    # Based on knowledge, 16 is achievable
    best_result = _backtrack(22, deadline)
    
    if not best_result:
        # Fallback: return the seed result
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return best_result
# EVOLVE-BLOCK-END
