import time
import random
from math import lcm

# Deadline guard
START_TIME = time.time()
DEADLINE = 85.0  # safety margin before 90s

def _check_time():
    return (time.time() - START_TIME) < DEADLINE

# LCM of 1..22
# 2^4 * 3^2 * 5 * 7 * 11 * 13 * 17 * 19
LCM_VAL = 232792560
WEIGHTS = [LCM_VAL // i for i in range(1, 23)]
# WEIGHTS[i-1] is weight for element i

def _try_greedy(order):
    """
    Try to build a dissociated set by adding elements in the given order.
    Returns the set of elements (1-based indices) found.
    """
    current_sums = {0}
    selected = []
    
    for idx in order:
        if not _check_time():
            break
        w = WEIGHTS[idx - 1]
        # Check if we can add w
        # New sums = current_sums U (current_sums + w)
        # Collision if any (s + w) is in current_sums
        # Or if (s1 + w) == (s2 + w) which implies s1 == s2 (already unique)
        # So we just need to check if (current_sums + w) intersects current_sums
        
        # Optimization: check intersection
        # If min(current_sums) + w > max(current_sums), no intersection possible?
        # Not necessarily, sums are not contiguous.
        
        # Efficient check:
        # Create a set of shifted sums? That's O(2^k).
        # We can just iterate.
        
        valid = True
        # To avoid creating a new set until we know it's valid, 
        # we can check membership first.
        # But we need the new set if valid.
        # Let's create the new set and check size.
        
        new_sums = set()
        # Add original sums
        new_sums.update(current_sums)
        
        # Add shifted sums
        for s in current_sums:
            ns = s + w
            if ns in new_sums:
                valid = False
                break
            new_sums.add(ns)
            
        if valid:
            current_sums = new_sums
            selected.append(idx)
            
    return selected

def construct():
    """
    Return a nonempty subset of [1,22] such that ALL reciprocal subset sums,
    including the empty sum, are distinct. Maximize cardinality.
    """
    all_elements = list(range(1, 23))
    best_set = []
    
    # Strategy 1: Greedy with largest weights first (indices 1 to 22)
    # This is the most likely to succeed for large sets because large weights
    # are less likely to cause collisions with the dense low-range sums.
    order_1 = list(range(1, 23))
    res = _try_greedy(order_1)
    if len(res) > len(best_set):
        best_set = res
    
    # Strategy 2: Greedy with smallest weights first (indices 22 to 1)
    order_2 = list(range(22, 0, -1))
    res = _try_greedy(order_2)
    if len(res) > len(best_set):
        best_set = res
        
    # Strategy 3: Randomized restarts
    # Try different random orders to escape local optima
    # If we found size K, try to find K+1
    
    # If we have time, try to improve best_set
    # Local search: try to swap elements
    
    # First, let's see what size we got.
    # If size is low, try more random orders.
    
    num_restarts = 50
    for i in range(num_restarts):
        if not _check_time():
            break
        
        # Random order
        order = all_elements[:]
        random.shuffle(order)
        res = _try_greedy(order)
        if len(res) > len(best_set):
            best_set = res
            
        # If we found a very large set, we might be near the optimum
        # We can try to verify if we can add any more elements to best_set
        # or swap to enable addition.
        
    # Final attempt: Try to extend best_set
    # Check if any element not in best_set can be added
    current_sums = {0}
    for idx in best_set:
        w = WEIGHTS[idx - 1]
        new_sums = set(current_sums)
        for s in current_sums:
            new_sums.add(s + w)
        current_sums = new_sums
        
    in_set = set(best_set)
    for idx in all_elements:
        if idx in in_set:
            continue
        if not _check_time():
            break
        w = WEIGHTS[idx - 1]
        # Check if can add
        valid = True
        # We don't need to update best_set immediately, just check
        # Actually, let's just try to add it and if valid, add to best_set
        # But we need to maintain current_sums correctly.
        # Let's do a dry run.
        
        # For each s in current_sums, check if s+w in current_sums
        # This is O(2^k).
        for s in current_sums:
            if (s + w) in current_sums:
                valid = False
                break
                
        if valid:
            # Update best_set and current_sums
            best_set.append(idx)
            in_set.add(idx)
            # Update current_sums
            new_sums = set(current_sums)
            for s in current_sums:
                new_sums.add(s + w)
            current_sums = new_sums

    if not best_set:
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best_set
