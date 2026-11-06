"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from functools import reduce

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85

_L = reduce(lcm, range(1, 23))
_INT_VALS = [_L // a for a in range(1, 23)]

def _try_order(order, best_size_target):
    """Try backtracking with given element order. Return best found."""
    best = []
    best_size = 0
    found = [False]
    
    def bt(pos, sel_vals, sel_sums):
        nonlocal best, best_size
        if found[0]:
            return
        if time.time() > _DEADLINE:
            return
        
        if len(sel_vals) + (22 - pos) <= best_size:
            return
        
        idx = order[pos]
        v = _INT_VALS[idx]
        new_sums = set()
        ok = True
        for s in sel_sums:
            ns = s + v
            if ns in sel_sums or ns in new_sums:
                ok = False
                break
            new_sums.add(ns)
        
        if ok:
            new_sel_sums = sel_sums | new_sums
            if len(sel_vals) + 1 > best_size:
                best = sel_vals + [v]
                best_size = len(best)
                if best_size >= best_size_target:
                    found[0] = True
                    return
            bt(pos + 1, sel_vals + [v], new_sel_sums)
        
        bt(pos + 1, sel_vals, sel_sums)
    
    bt(0, [], {0})
    return best, best_size

def _search():
    """Try multiple orderings to find max cardinality set."""
    # Try natural order first
    best_result = None
    best_size = 0
    
    # Try to find size 17, 18, etc.
    for target in [18, 17, 16, 15]:
        # Try several orderings
        orderings = [
            list(range(22)),                    # natural
            list(range(22, 0, -1)),            # reverse
            list(range(0, 22, 2)) + list(range(1, 22, 2)),  # evens then odds
            list(range(1, 22, 2)) + list(range(0, 22, 2)),  # odds then evens
        ]
        
        for order in orderings:
            if time.time() > _DEADLINE:
                break
            res, size = _try_order(order, target)
            if size > best_size:
                best_size = size
                best_result = res
            if size >= target:
                break
        if best_size >= target:
            break
    
    if best_result:
        return [_L // v for v in best_result]
    return None

def _verify(result):
    if not result:
        return False
    vals = [_L // a for a in result]
    sums = {0}
    for v in vals:
        new_sums = set()
        for s in sums:
            ns = s + v
            if ns in sums or ns in new_sums:
                return False
            new_sums.add(ns)
        sums |= new_sums
    return True

def construct():
    result = _search()
    if result and _verify(result):
        return result
    fallback = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    if _verify(fallback):
        return fallback
    return [1]
# EVOLVE-BLOCK-END
