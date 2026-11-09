import numpy as np
import random

def _can_add(arr, sums_set, x):
    """Check if x can be added to arr without violating unique triple sums."""
    if not arr:
        return True, np.array([x+x+x]) # Wait, if arr is empty, only x,x,x is generated? No, we add x to empty set.
        # If arr is empty, new sums are just x+x+x? No, we are adding x to the set.
        # The set becomes {x}. Triple sums of {x} is {3x}.
        # So yes, new_sums = [3x].
        
    n = len(arr)
    A = np.array(arr)
    
    s1 = x + x + x
    s2 = x + x + A
    s3 = x + A[:, None] + A[None, :]
    s3_flat = s3.ravel()
    
    all_new_sums = np.concatenate(([s1], s2, s3_flat))
    
    for s in all_new_sums:
        if s in sums_set:
            return False, None
            
    return True, all_new_sums

def _greedy_with_order(order, max_val=4096, target_size=80):
    arr = []
    sums_set = set()
    
    for x in order:
        if x > max_val:
            continue
        if len(arr) >= target_size:
            break
            
        valid, new_sums = _can_add(arr, sums_set, x)
        
        if valid:
            arr.append(x)
            sums_set.update(new_sums)
            
    return arr

def construct():
    """Construct a set of up to 80 distinct positive integers <= 4096 with unique triple sums."""
    max_val = 4096
    target_size = 80
    
    # Try standard ascending order
    order1 = list(range(1, max_val + 1))
    best_arr = _greedy_with_order(order1, max_val, target_size)
    
    if len(best_arr) < target_size:
        # Try random orders
        for _ in range(10):
            random_order = list(range(1, max_val + 1))
            random.shuffle(random_order)
            current_arr = _greedy_with_order(random_order, max_val, target_size)
            if len(current_arr) > len(best_arr):
                best_arr = current_arr
                if len(best_arr) == target_size:
                    break
                    
    return best_arr
