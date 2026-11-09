import time
import random
from itertools import combinations

def _is_sunflower_triple(a, b, c):
    """Check if three sets form a sunflower (all pairwise intersections equal)."""
    return a & b == a & c == b & c

def construct():
    deadline = time.time() + 85  # Safety margin
    
    # Pre-compute all 4-element subsets of [1,16] as bitmasks for speed
    # and also keep the original tuples for output
    all_sets_info = []
    for row in combinations(range(1, 17), 4):
        mask = 0
        for i in row:
            mask |= (1 << (i-1))
        all_sets_info.append((mask, row))
    
    n_total = len(all_sets_info)
    all_masks = [info[0] for info in all_sets_info]
    all_rows = [info[1] for info in all_sets_info]
    
    # Helper: check if mask can be added to a set of chosen masks
    # Optimized for speed
    def can_add_mask(mask, chosen_masks):
        n = len(chosen_masks)
        for i in range(n):
            mi = chosen_masks[i]
            mi_mask = mi & mask
            for j in range(i + 1, n):
                mj = chosen_masks[j]
                inter = mi & mj
                if mi_mask == inter and (mj & mask) == inter:
                    return False
        return True
    
    # Helper: run greedy given an order of indices
    def greedy_pass(order):
        chosen_masks = []
        chosen_indices = []
        for idx in order:
            if time.time() > deadline:
                break
            mask = all_masks[idx]
            if can_add_mask(mask, chosen_masks):
                chosen_masks.append(mask)
                chosen_indices.append(idx)
        return chosen_indices
    
    # Local search: try to improve by swapping one chosen set with one unchosen set
    def local_search(chosen_indices, max_iters=50):
        chosen_set = set(chosen_indices)
        chosen_masks = [all_masks[i] for i in chosen_indices]
        unchosen = [i for i in range(n_total) if i not in chosen_set]
        
        improved = True
        iterations = 0
        while improved and iterations < max_iters:
            improved = False
            iterations += 1
            if time.time() > deadline - 5:
                break
            
            # Try each swap: remove one chosen, add one unchosen
            for out_idx in range(len(chosen_indices)):
                if time.time() > deadline - 5:
                    break
                out_mask = chosen_masks[out_idx]
                # Temporarily remove out_idx
                # Using slice is O(M), which is fine for M < 300
                temp_chosen_masks = chosen_masks[:out_idx] + chosen_masks[out_idx+1:]
                
                for in_idx in unchosen:
                    in_mask = all_masks[in_idx]
                    if can_add_mask(in_mask, temp_chosen_masks):
                        # Perform the swap
                        chosen_indices[out_idx] = in_idx
                        chosen_masks[out_idx] = in_mask
                        chosen_set.discard(chosen_indices[out_idx])
                        chosen_set.add(in_idx)
                        # Update unchosen list
                        # List removal is O(N), but N is 1820, so it's okay
                        unchosen.remove(in_idx)
                        unchosen.append(out_idx)
                        improved = True
                        break
                
                if improved:
                    break
        
        return chosen_indices
    
    best_family = []
    best_size = 0
    
    indices = list(range(n_total))
    
    # We will run multiple random restarts
    # This aligns with the "randomized node ordering" strategy for MIS heuristics
    
    while time.time() < deadline - 10:
        random.shuffle(indices)
        family_indices = greedy_pass(indices)
        
        # If this greedy result is already better, polish it
        if len(family_indices) > best_size:
            # Apply local search if we have time
            if time.time() < deadline - 20:
                family_indices = local_search(family_indices)
            
            if len(family_indices) > best_size:
                best_size = len(family_indices)
                best_family = [all_rows[i] for i in family_indices]
                
        # If we find a very large family, we can try to polish it further
        # by running local search on the best_family directly if time permits
        if best_size > 150 and time.time() < deadline - 30:
             # Re-run local search on the current best to see if we can squeeze more
             # This is a "polish pass"
             polished = local_search([i for i, r in enumerate(all_rows) if r in best_family], max_iters=20)
             if len(polished) > best_size:
                 best_size = len(polished)
                 best_family = [all_rows[i] for i in polished]

    return [sorted(s) for s in best_family]
