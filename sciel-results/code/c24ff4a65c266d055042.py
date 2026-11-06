"""Rewritten program: fast iterative random greedy restarts for dissociated reciprocal sets using integer arithmetic."""
import time
import random

# EVOLVE-BLOCK-START

# LCM of 1..22 is 232792560
_LCM = 232792560
_INV_INT = {x: _LCM // x for x in range(1, 23)}

def _is_dissociated_int(elements):
    """
    Check if all subset sums of 1/x for x in elements are distinct.
    Uses integer arithmetic based on LCM(1..22).
    """
    if not elements:
        return True
    sums = {0}
    for x in elements:
        inv = _INV_INT[x]
        new_sums = set()
        for s in sums:
            new_sums.add(s + inv)
        # Collision if any new sum was already present
        if sums & new_sums:
            return False
        sums.update(new_sums)
    return True

def construct():
    start_time = time.time()
    # Use 55s out of 60s for safety
    deadline = start_time + 55
    
    all_elements = list(range(1, 23))
    
    best_set = set()
    best_size = 0
    
    # Initial greedy pass to get a decent starting point
    for _ in range(20):
        perm = all_elements[:]
        random.shuffle(perm)
        current_set = set()
        for x in perm:
            if _is_dissociated_int(current_set | {x}):
                current_set.add(x)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
    if best_size == 0:
        best_set = {1}
        best_size = 1

    # Random greedy restarts
    while time.time() < deadline:
        perm = all_elements[:]
        random.shuffle(perm)
        
        current_set = set()
        for x in perm:
            if time.time() > deadline - 1:
                break
            if _is_dissociated_int(current_set | {x}):
                current_set.add(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
        # Local search: try to add more elements to the current set
        if len(current_set) >= best_size and time.time() < deadline - 2:
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                candidates = [x for x in all_elements if x not in current_set]
                random.shuffle(candidates)
                for x in candidates:
                    if time.time() > deadline - 2:
                        break
                    if _is_dissociated_int(current_set | {x}):
                        current_set.add(x)
                        improved = True
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = set(current_set)
                        break

    return sorted(best_set)

# EVOLVE-BLOCK-END
