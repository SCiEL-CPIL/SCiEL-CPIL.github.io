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

def _forms_sunflower(a, b, c):
    # Check if three sets form a sunflower: all three pairwise intersections are equal as sets
    ab = a & b
    ac = a & c
    bc = b & c
    return ab == ac == bc

def _greedy_with_order(all_sets, all_masks, order, deadline):
    chosen = []
    chosen_masks = []
    for idx in order:
        if time.time() >= deadline:
            break
        m = all_masks[idx]
        ok = True
        for i in range(len(chosen)):
            if time.time() >= deadline:
                break
            for j in range(i + 1, len(chosen)):
                if _forms_sunflower(chosen_masks[i], chosen_masks[j], m):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            chosen.append(all_sets[idx])
            chosen_masks.append(m)
    return chosen

def _local_search(family, all_sets, all_masks, n_sets, deadline):
    """
    Try to add sets that don't conflict with the current family.
    Also includes a swap mechanism to escape local optima.
    """
    current = list(family)
    current_masks = [_to_mask(s) for s in current]
    current_mask_set = set(current_masks)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try to add sets (standard local search)
        added = True
        while added and time.time() < deadline:
            added = False
            for idx in range(n_sets):
                if time.time() >= deadline:
                    break
                m = all_masks[idx]
                if m in current_mask_set:
                    continue
                
                # Check if adding s creates a sunflower with any pair in current
                ok = True
                for i in range(len(current)):
                    for j in range(i + 1, len(current)):
                        if _forms_sunflower(current_masks[i], current_masks[j], m):
                            ok = False
                            break
                    if not ok:
                        break
                
                if ok:
                    current.append(all_sets[idx])
                    current_masks.append(m)
                    current_mask_set.add(m)
                    added = True
                    improved = True
        
        # 2. If no sets could be added, try swaps
        if not added and time.time() < deadline:
            swap_found = False
            # Limit swap attempts to avoid timeout
            for _ in range(50):
                if time.time() >= deadline:
                    break
                if not current:
                    break
                
                # Pick a random set to remove
                rem_idx = random.randint(0, len(current) - 1)
                rem_set = current[rem_idx]
                rem_mask = current_masks[rem_idx]
                
                # Remove it
                current.pop(rem_idx)
                current_masks.pop(rem_idx)
                current_mask_set.remove(rem_mask)
                
                # Try to add sets greedily to the reduced family
                # We use a simple greedy pass over all available sets
                # To speed up, we can shuffle the order
                order = list(range(n_sets))
                random.shuffle(order)
                
                added_in_swap = 0
                for idx in order:
                    if time.time() >= deadline:
                        break
                    m = all_masks[idx]
                    if m in current_mask_set:
                        continue
                    
                    ok = True
                    for i in range(len(current)):
                        for j in range(i + 1, len(current)):
                            if _forms_sunflower(current_masks[i], current_masks[j], m):
                                ok = False
                                break
                        if not ok:
                            break
                    
                    if ok:
                        current.append(all_sets[idx])
                        current_masks.append(m)
                        current_mask_set.add(m)
                        added_in_swap += 1
                
                # Net gain
                if added_in_swap > 1:
                    improved = True
                    swap_found = True
                    break # Restart the main loop to try more adds
                elif added_in_swap == 1:
                    # Neutral move, might help escape, but let's be conservative
                    # Or accept if we want to explore?
                    # For now, only accept if it improves size.
                    pass
                else:
                    # Restore
                    current.append(rem_set)
                    current_masks.append(rem_mask)
                    current_mask_set.add(rem_mask)
                    # If we tried 50 swaps and none improved, we are likely stuck
                    # We can break the swap loop
                    if _ == 49:
                         break
            if swap_found:
                continue # Go back to adding phase

    return current

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    all_sets = [set(row) for row in combinations(range(1, 17), 4)]
    all_masks = [_to_mask(s) for s in all_sets]
    n_sets = len(all_sets)
    
    best = []
    
    # First pass: original order
    result = _greedy_with_order(all_sets, all_masks, list(range(n_sets)), deadline)
    if len(result) > len(best):
        best = result
    
    # Multiple randomized restarts
    for restart in range(200):
        if time.time() >= deadline:
            break
        order = list(range(n_sets))
        random.shuffle(order)
        result = _greedy_with_order(all_sets, all_masks, order, deadline)
        if len(result) > len(best):
            best = result
            # Try local search to extend the best family found so far
            if time.time() < deadline:
                extended = _local_search(best, all_sets, all_masks, n_sets, deadline)
                if len(extended) > len(best):
                    best = extended
    
    return [sorted(s) for s in best]
# EVOLVE-BLOCK-END
