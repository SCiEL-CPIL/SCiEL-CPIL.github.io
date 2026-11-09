"""Find maximum cardinality dissociated reciprocal subset of [1,22]."""
import time
from fractions import Fraction
from itertools import combinations
import random

# EVOLVE-BLOCK-START
import time

# LCM of 1..22
_LCM_22 = 232792560

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # Precompute scaled reciprocals: 1/i -> LCM // i
    recips = [_LCM_22 // i for i in range(1, 23)]
    
    best_set = []
    
    def dfs(idx, current_set, sums_set):
        nonlocal best_set
        
        # Time check
        if time.time() > deadline:
            return
        
        # Update best solution
        if len(current_set) > len(best_set):
            best_set = list(current_set)
            
        # Pruning: if even adding all remaining elements doesn't beat best, stop
        if len(current_set) + (22 - idx) <= len(best_set):
            return
            
        # Iterate over remaining elements
        for i in range(idx, 22):
            if time.time() > deadline:
                return
            
            r = recips[i]
            new_sums = set()
            conflict = False
            
            # Check if adding r creates a collision in subset sums
            for s in sums_set:
                ns = s + r
                if ns in sums_set:
                    conflict = True
                    break
                new_sums.add(ns)
                
            if conflict:
                continue
                
            # Recurse with new sums
            next_sums = sums_set.union(new_sums)
            dfs(i + 1, current_set + [i + 1], next_sums)
            
    dfs(0, [], {0})
    
    return best_set

# EVOLVE-BLOCK-END
