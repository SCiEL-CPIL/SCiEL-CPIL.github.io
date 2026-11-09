# EVOLVE-BLOCK-START
import numpy as np

def _init_sum_set(arr):
    """Initialize the set of all triple sums for a given array."""
    n = len(arr)
    sums = set()
    for i in range(n):
        ai = arr[i]
        for j in range(i, n):
            aj = arr[j]
            for k in range(j, n):
                ak = arr[k]
                sums.add(ai + aj + ak)
    return sums

def _check_add(arr, sums_set, candidate):
    """
    Check if adding candidate to arr keeps triple sums unique.
    Returns (True, updated_sums_set) if valid, (False, None) otherwise.
    Modifies sums_set in place if valid.
    """
    n = len(arr)
    c = candidate
    
    # 1. c + c + c
    s1 = 3 * c
    if s1 in sums_set:
        return False, None
    new_sums = {s1}
    
    # 2. c + c + a_i
    for i in range(n):
        s = 2 * c + arr[i]
        if s in sums_set or s in new_sums:
            return False, None
        new_sums.add(s)
        
    # 3. c + a_i + a_j (i <= j)
    for i in range(n):
        ai = arr[i]
        for j in range(i, n):
            aj = arr[j]
            s = c + ai + aj
            if s in sums_set or s in new_sums:
                return False, None
            new_sums.add(s)
            
    # If we get here, it's valid.
    sums_set.update(new_sums)
    return True, sums_set

def _greedy_from(start_set, max_val=4096, target_size=80):
    """Greedily add elements to start_set."""
    selected = list(start_set)
    selected_set = set(selected)
    current_sums = _init_sum_set(selected)
    
    for x in range(1, max_val + 1):
        if len(selected) >= target_size:
            break
        
        if x in selected_set:
            continue
            
        valid, current_sums = _check_add(selected, current_sums, x)
        if valid:
            selected.append(x)
            selected_set.add(x)
            
    return selected

def construct():
    """Construct a set of up to 80 distinct positive integers <= 4096 with unique triple sums."""
    target_size = 80
    max_val = 4096
    
    # Phase 1: Greedy from empty set
    selected = _greedy_from([], max_val, target_size)
    best_set = list(selected)
    best_size = len(best_set)
    
    # Phase 2: Local Search
    # Try to improve by removing a large element and refilling.
    # We'll try removing the largest K elements and refilling.
    
    for attempt in range(20):
        if best_size >= target_size:
            break
            
        # Remove the largest element
        if len(best_set) == 0:
            break
            
        remove_idx = len(best_set) - 1
        removed_val = best_set[remove_idx]
        temp_set = best_set[:remove_idx] + best_set[remove_idx+1:]
        
        # Refill
        new_set = _greedy_from(temp_set, max_val, target_size)
        
        if len(new_set) > best_size:
            best_set = new_set
            best_size = len(best_set)
        else:
            # If we didn't improve, maybe try removing the second largest?
            # Or just break if we are stuck?
            # To avoid infinite loops or slow progress, we can try a few more specific swaps.
            # For now, if the largest swap didn't help, it's likely stuck.
            # But let's try removing the largest 2?
            # Actually, if removing the largest didn't increase size, 
            # the greedy refill probably just put the same or similar numbers back.
            # Let's try removing the largest element and the second largest.
            if len(best_set) >= 2:
                 remove_idx2 = len(best_set) - 2
                 temp_set2 = best_set[:remove_idx2] + best_set[remove_idx2+1:]
                 new_set2 = _greedy_from(temp_set2, max_val, target_size)
                 if len(new_set2) > best_size:
                     best_set = new_set2
                     best_size = len(best_set)
            # If still no improvement, break to save time
            break

    return best_set
# EVOLVE-BLOCK-END
