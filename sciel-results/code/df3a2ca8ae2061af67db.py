"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _to_mask(s):
    m = 0
    for x in s:
        m |= (1 << (x - 1))
    return m

def _mask_to_set(m):
    return [i + 1 for i in range(16) if m & (1 << i)]

def _forms_sunflower(a, b, c):
    # Check if three sets form a sunflower: all three pairwise intersections are equal as sets
    ab = a & b
    ac = a & c
    bc = b & c
    return ab == ac == bc

def _greedy_with_order(all_sets, order, deadline):
    chosen = []
    chosen_masks = []
    for idx in order:
        if time.time() >= deadline:
            break
        s = all_sets[idx]
        m = _to_mask(s)
        ok = True
        for i in range(len(chosen)):
            if time.time() >= deadline:
                break
            for j in range(i + 1, len(chosen)):
                if _forms_sunflower(chosen[i], chosen[j], s):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            chosen.append(s)
            chosen_masks.append(m)
    return chosen

def _local_search(family, all_sets, n_sets, deadline):
    """
    Try to add sets that don't conflict with the current family.
    """
    current = list(family)
    current_masks = [_to_mask(s) for s in current]
    
    # Build a set of masks for fast lookup
    current_mask_set = set(current_masks)
    
    # Try to add sets from all_sets that are not already in the family
    added = True
    while added and time.time() < deadline:
        added = False
        for idx in range(n_sets):
            if time.time() >= deadline:
                break
            s = all_sets[idx]
            m = _to_mask(s)
            if m in current_mask_set:
                continue
            
            # Check if adding s creates a sunflower with any pair in current
            ok = True
            for i in range(len(current)):
                for j in range(i + 1, len(current)):
                    if _forms_sunflower(current[i], current[j], s):
                        ok = False
                        break
                if not ok:
                    break
            
            if ok:
                current.append(s)
                current_masks.append(m)
                current_mask_set.add(m)
                added = True
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    all_sets = [set(row) for row in combinations(range(1, 17), 4)]
    n_sets = len(all_sets)
    
    best = []
    
    # First pass: original order
    result = _greedy_with_order(all_sets, list(range(n_sets)), deadline)
    if len(result) > len(best):
        best = result
    
    # Multiple randomized restarts
    for restart in range(200):
        if time.time() >= deadline:
            break
        order = list(range(n_sets))
        random.shuffle(order)
        result = _greedy_with_order(all_sets, order, deadline)
        if len(result) > len(best):
            best = result
            # Try local search to extend the best family found so far
            if time.time() < deadline:
                extended = _local_search(best, all_sets, n_sets, deadline)
                if len(extended) > len(best):
                    best = extended
    
    return [sorted(s) for s in best]
# EVOLVE-BLOCK-END
