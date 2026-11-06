"""Improved B_3[1] construction: optimized greedy with local search."""
# EVOLVE-BLOCK-START
import time
import random

def _can_add(x, selected, existing_sums):
    """Check if x can be added to selected without violating B_3[1] property."""
    n = len(selected)
    
    # (x, x, x)
    s = 3 * x
    if s in existing_sums:
        return False
    
    # (x, x, y) for y in selected
    for y in selected:
        s = 2 * x + y
        if s in existing_sums:
            return False
    
    # (x, y, y) for y in selected
    for y in selected:
        s = x + 2 * y
        if s in existing_sums:
            return False
    
    # (x, y, z) for y <= z in selected
    for i in range(n):
        y = selected[i]
        for j in range(i, n):
            z = selected[j]
            s = x + y + z
            if s in existing_sums:
                return False
    
    return True

def _add_element(x, selected, existing_sums):
    """Add x to selected and update existing_sums."""
    n = len(selected)
    
    # (x, x, x)
    existing_sums.add(3 * x)
    
    # (x, x, y) for y in selected
    for y in selected:
        existing_sums.add(2 * x + y)
    
    # (x, y, y) for y in selected
    for y in selected:
        existing_sums.add(x + 2 * y)
    
    # (x, y, z) for y <= z in selected
    for i in range(n):
        y = selected[i]
        for j in range(i, n):
            z = selected[j]
            existing_sums.add(x + y + z)
    
    selected.append(x)

def _greedy_build(order, deadline):
    """Build a B_3[1] set by greedily adding elements in the given order."""
    selected = []
    existing_sums = set()

    for x in order:
        if time.time() > deadline:
            break
        if len(selected) >= 80:
            break
        if _can_add(x, selected, existing_sums):
            _add_element(x, selected, existing_sums)

    return selected

def _local_search(selected, deadline):
    """Try to improve the set by swapping elements."""
    selected = list(selected)
    existing_sums = set()
    
    # Rebuild existing_sums
    n = len(selected)
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                existing_sums.add(selected[i] + selected[j] + selected[k])
    
    best_size = len(selected)
    
    # Try removing one element and adding a better one
    for remove_idx in range(n):
        if time.time() > deadline:
            break
        
        x_remove = selected[remove_idx]
        
        # Temporarily remove
        selected.remove(x_remove)
        n = len(selected)
        
        # Rebuild sums without x_remove
        existing_sums = set()
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    existing_sums.add(selected[i] + selected[j] + selected[k])
        
        # Try adding candidates
        added = False
        for x_add in range(1, 4097):
            if x_add == x_remove:
                continue
            if x_add in selected:
                continue
            if _can_add(x_add, selected, existing_sums):
                _add_element(x_add, selected, existing_sums)
                added = True
                break
        
        if added:
            if len(selected) > best_size:
                best_size = len(selected)
        else:
            # Put back
            _add_element(x_remove, selected, existing_sums)
            selected.sort()
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    best = []
    best_size = 0
    
    # Strategy 1: increasing order
    result = _greedy_build(range(1, 4097), deadline)
    if len(result) > best_size:
        best = result
        best_size = len(result)
    
    # Strategy 2: multiple random restarts with structured orderings
    candidates = list(range(1, 4097))
    
    # First: try some structured orderings
    orderings = [
        list(range(1, 4097)),           # increasing
        list(range(4096, 0, -1)),       # decreasing
        [i for i in range(1, 4097) if i % 2 == 1],  # odd only
        [i for i in range(1, 4097) if i % 3 != 0],  # not divisible by 3
    ]
    
    for order in orderings:
        if time.time() > deadline - 2.0:
            break
        result = _greedy_build(order, deadline)
        if len(result) > best_size:
            best = result
            best_size = len(result)
    
    # Strategy 3: random restarts
    num_restarts = 30
    for _ in range(num_restarts):
        if time.time() > deadline - 3.0:
            break
        random.shuffle(candidates)
        result = _greedy_build(candidates, deadline)
        if len(result) > best_size:
            best = result
            best_size = len(result)
    
    # Strategy 4: local search on best result
    if best_size > 0 and time.time() < deadline - 5.0:
        improved = _local_search(best, deadline)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    return best
# EVOLVE-BLOCK-END
