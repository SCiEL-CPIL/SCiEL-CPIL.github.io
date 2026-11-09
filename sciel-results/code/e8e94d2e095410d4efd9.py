import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 16
    k = 4
    max_size = 250
    
    # Generate all possible 4-element subsets of [1..16] as bitmasks
    # and keep the original tuples for output
    all_sets_info = []
    for row in combinations(range(1, n + 1), k):
        mask = 0
        for i in row:
            mask |= (1 << (i - 1))
        all_sets_info.append((mask, row))
    
    total = len(all_sets_info)  # 1820
    all_masks = [info[0] for info in all_sets_info]
    all_rows = [info[1] for info in all_sets_info]
    
    def can_add(new_mask, family_masks):
        """
        Check if new_mask can be added to family_masks without creating a sunflower.
        A sunflower is formed if there exist A, B in family_masks such that
        A & B == A & new_mask == B & new_mask.
        """
        fm = family_masks
        L = len(fm)
        for i in range(L):
            ai = fm[i]
            inter_ai_new = ai & new_mask
            # Optimization: If inter_ai_new is not a subset of ai & bj for some j, 
            # we can't just skip, we must check equality.
            for j in range(i + 1, L):
                bj = fm[j]
                # Check sunflower condition
                if (ai & bj) == inter_ai_new and (bj & new_mask) == inter_ai_new:
                    return False
        return True

    def greedy_build(order):
        """Build a family using a specific order of indices."""
        family_masks = []
        family_indices = []
        for idx in order:
            if len(family_masks) >= max_size:
                break
            mask = all_masks[idx]
            if can_add(mask, family_masks):
                family_masks.append(mask)
                family_indices.append(idx)
        return family_masks, family_indices

    def local_search(family_masks, family_indices, time_limit):
        """
        Improve the family via 1-for-1 and 2-for-1 swaps.
        """
        current_masks = family_masks[:]
        current_indices = family_indices[:]
        current_set = set(current_indices)
        unchosen_indices = [i for i in range(total) if i not in current_set]
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Try 1-for-1 swaps
            # Remove a random set, try to add a random unchosen set
            if unchosen_indices:
                for _ in range(100): # Limit attempts per pass to avoid stalling
                    if time.time() > time_limit:
                        break
                    if not current_masks:
                        break
                    
                    rem_pos = random.randint(0, len(current_masks) - 1)
                    rem_mask = current_masks[rem_pos]
                    rem_idx = current_indices[rem_pos]
                    
                    # Temporarily remove
                    temp_masks = current_masks[:rem_pos] + current_masks[rem_pos+1:]
                    temp_indices = current_indices[:rem_pos] + current_indices[rem_pos+1:]
                    
                    # Try to add a random unchosen set
                    added = False
                    # Try a few random candidates
                    candidates = random.sample(unchosen_indices, min(50, len(unchosen_indices)))
                    for add_idx in candidates:
                        add_mask = all_masks[add_idx]
                        if can_add(add_mask, temp_masks):
                            # Perform swap
                            current_masks[rem_pos] = add_mask
                            current_indices[rem_pos] = add_idx
                            current_set.discard(rem_idx)
                            current_set.add(add_idx)
                            unchosen_indices.remove(add_idx)
                            unchosen_indices.append(rem_idx)
                            improved = True
                            added = True
                            break
                    # If no swap found, we don't revert anything because we didn't modify current_masks yet
                    # Wait, we only modify current_masks if added is True.
                    # If added is False, current_masks is unchanged. This is correct.
                    if added:
                        break # Restart the loop with the new state

            # 2. Try 2-for-1 swaps (Remove 1, Add 2)
            # This is more powerful for increasing size
            if len(unchosen_indices) >= 2:
                for _ in range(50):
                    if time.time() > time_limit:
                        break
                    if not current_masks:
                        break
                    
                    rem_pos = random.randint(0, len(current_masks) - 1)
                    rem_mask = current_masks[rem_pos]
                    rem_idx = current_indices[rem_pos]
                    
                    # Temporarily remove
                    temp_masks = current_masks[:rem_pos] + current_masks[rem_pos+1:]
                    temp_indices = current_indices[:rem_pos] + current_indices[rem_pos+1:]
                    
                    # Try to add two sets
                    # Pick two random unchosen sets
                    add_indices = random.sample(unchosen_indices, min(100, len(unchosen_indices)))
                    # We need to find a pair (C, D) that can both be added
                    # Try adding C first, then D
                    found_pair = None
                    for i in range(len(add_indices)):
                        c_idx = add_indices[i]
                        c_mask = all_masks[c_idx]
                        if not can_add(c_mask, temp_masks):
                            continue
                        
                        # Tentatively add C
                        temp_masks_with_c = temp_masks + [c_mask]
                        
                        for j in range(len(add_indices)):
                            if i == j:
                                continue
                            d_idx = add_indices[j]
                            d_mask = all_masks[d_idx]
                            if d_idx == c_idx:
                                continue
                            if not can_add(d_mask, temp_masks_with_c):
                                continue
                            
                            # Found a valid pair (C, D)
                            found_pair = (c_idx, d_idx)
                            break
                        if found_pair:
                            break
                    
                    if found_pair:
                        c_idx, d_idx = found_pair
                        c_mask = all_masks[c_idx]
                        d_mask = all_masks[d_idx]
                        
                        # Perform the 2-for-1 swap
                        # Remove rem_idx
                        current_masks.pop(rem_pos)
                        current_indices.pop(rem_pos)
                        current_set.discard(rem_idx)
                        unchosen_indices.remove(c_idx)
                        unchosen_indices.remove(d_idx)
                        unchosen_indices.append(rem_idx)
                        
                        # Add c_idx and d_idx
                        current_masks.append(c_mask)
                        current_indices.append(c_idx)
                        current_set.add(c_idx)
                        
                        current_masks.append(d_mask)
                        current_indices.append(d_idx)
                        current_set.add(d_idx)
                        
                        improved = True
                        break # Restart loop

        return current_masks, current_indices

    best_family_masks = []
    best_family_indices = []
    best_size = 0
    
    # Strategy: Run multiple greedy passes with different orderings, then local search
    indices = list(range(total))
    
    # Define a few deterministic orders to ensure variety
    orders = []
    
    # 1. Lexicographic
    orders.append(indices[:])
    
    # 2. Reverse Lexicographic
    orders.append(indices[::-1])
    
    # 3. Sorted by mask value
    orders.append(sorted(indices, key=lambda i: all_masks[i]))
    
    # 4. Sorted by popcount (all are 4, so this is same as mask value essentially, skip)
    
    # Add random orders
    for _ in range(5):
        random.shuffle(indices)
        orders.append(indices[:])
    
    search_time_limit = deadline - 5.0
    
    for order in orders:
        if time.time() > search_time_limit:
            break
            
        # Greedy Build
        fam_masks, fam_indices = greedy_build(order)
        
        # Local Search
        if time.time() < search_time_limit:
            fam_masks, fam_indices = local_search(fam_masks, fam_indices, search_time_limit)
            
        if len(fam_masks) > best_size:
            best_size = len(fam_masks)
            best_family_masks = fam_masks[:]
            best_family_indices = fam_indices[:]
            
        if best_size >= max_size:
            break
            
    # Final Verification
    # Ensure the returned family is valid
    verified_masks = []
    for mask in best_family_masks:
        if can_add(mask, verified_masks):
            verified_masks.append(mask)
            
    # Convert masks back to sorted lists
    # We need to map mask to row. 
    # Since mask is unique for each row, we can just use the row from all_sets_info if we had a map.
    # But we have best_family_indices.
    verified_rows = [all_rows[idx] for idx in best_family_indices if all_masks[idx] in verified_masks]
    
    # Wait, the verification loop above modifies verified_masks but not best_family_indices.
    # Let's re-verify properly.
    final_masks = []
    final_indices = []
    for idx in best_family_indices:
        mask = all_masks[idx]
        if can_add(mask, final_masks):
            final_masks.append(mask)
            final_indices.append(idx)
            
    return [sorted(all_rows[idx]) for idx in final_indices]

if __name__ == "__main__":
    res = construct()
    print(len(res))
