"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from functools import reduce

def _compute_lcm_1_22():
    return reduce(lcm, range(1, 23))

_L = _compute_lcm_1_22()

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    vals = {i: _L // i for i in range(1, 23)}
    best = []
    
    def backtrack(current, sums_set, start, time_limit):
        nonlocal best
        if time.time() > time_limit:
            return
        if len(current) > len(best):
            best = list(current)
        
        # Pruning: can't beat best if not enough elements remaining
        remaining = 22 - start + 1
        if len(current) + remaining <= len(best):
            return
        
        for i in range(start, 23):
            if time.time() > time_limit:
                return
            v = vals[i]
            new_sums = set()
            valid = True
            for s in sums_set:
                ns = s + v
                if ns in sums_set or ns in new_sums:
                    valid = False
                    break
                new_sums.add(ns)
            if not valid:
                continue
            backtrack(current + [i], sums_set | new_sums, i + 1, time_limit)
    
    backtrack([], {0}, 1, deadline)
    
    if not best:
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return best
# EVOLVE-BLOCK-END
