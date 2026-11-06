"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from functools import reduce

def _compute_lcm_1_22():
    return reduce(lcm, range(1, 23))

_L = _compute_lcm_1_22()

def _greedy_select(order):
    """
    Greedily select elements from the given order.
    Returns the list of selected elements.
    """
    current_sums = {0}
    selected = []
    for a in order:
        v = _L // a
        # Check for collision
        # If any s + v is in current_sums, collision
        # Note: s+v in current_sums implies s' = s+v, so s = s'-v.
        # This means subset sum s' - v = s.
        # Since v is not in current_sums (it's a new element), 
        # this corresponds to a non-trivial relation.
        
        # Efficient check
        for s in current_sums:
            if (s + v) in current_sums:
                break
        else:
            # No collision, accept
            selected.append(a)
            # Update sums
            new_sums = set()
            for s in current_sums:
                new_sums.add(s + v)
            current_sums.update(new_sums)
            
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Strategy 1: Largest denominators first (22 down to 1)
    order1 = list(range(22, 0, -1))
    res1 = _greedy_select(order1)
    
    if time.time() > deadline:
        return res1 if res1 else [1]
        
    # Strategy 2: Smallest denominators first (1 to 22)
    order2 = list(range(1, 23))
    res2 = _greedy_select(order2)
    
    if time.time() > deadline:
        best = res1 if len(res1) >= len(res2) else res2
        return best if best else [1]
        
    # Strategy 3: Randomized restarts if time permits
    best_res = res1 if len(res1) >= len(res2) else res2
    best_len = len(best_res)
    
    import random
    # Try a few random orders
    for _ in range(20):
        if time.time() > deadline - 5.0:
            break
        order3 = list(range(1, 23))
        random.shuffle(order3)
        res3 = _greedy_select(order3)
        if len(res3) > best_len:
            best_res = res3
            best_len = len(res3)
            
    return best_res if best_res else [1]

# EVOLVE-BLOCK-END
