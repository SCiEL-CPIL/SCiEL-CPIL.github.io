import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    n_points = 16
    # Generate all possible triples
    all_triples = []
    for a in range(n_points):
        for b in range(a+1, n_points):
            for c in range(b+1, n_points):
                all_triples.append((a, b, c))
                
    # Map triple to bitmask for fast subset check
    # triple_mask[t] = bitmask of points in t
    triple_masks = [0] * len(all_triples)
    for i, t in enumerate(all_triples):
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks[i] = mask

    # Conflict check: two triples conflict if they share 2 or more points
    # We can precompute conflicts or check on the fly
    # Since n_points is small, we can use a set of active triples
    
    def get_alpha(active_triples_indices):
        """
        Calculate independence number of the hypergraph defined by active_triples_indices.
        active_triples_indices is a list of indices into all_triples.
        """
        if not active_triples_indices:
            return n_points
        
        # For each subset of points, check if it contains any active triple
        # We want the largest subset that does NOT contain any active triple.
        # Iterate all 2^16 subsets.
        max_indep = 0
        # Precompute masks for active triples
        active_masks = [triple_masks[i] for i in active_triples_indices]
        
        # Optimization: If a subset contains an active triple, it's not independent.
        # We can iterate subsets by size descending? Or just all.
        # 2^16 = 65536. For each, check against ~37 triples.
        # 65536 * 37 ~ 2.4M ops.
        
        for mask in range(1 << n_points):
            popcount = mask.bit_count()
            if popcount <= max_indep:
                continue
            # Check if any active triple is a subset of mask
            is_independent = True
            for am in active_masks:
                if (mask & am) == am:
                    is_independent = False
                    break
            if is_independent:
                max_indep = popcount
                # Early stop if we found a set of size n_points? No, 0 is always independent.
                # Max possible is n_points.
                if max_indep == n_points:
                    break
                    
        return max_indep

    # Greedy initialization
    random.shuffle(all_triples)
    active_indices = []
    # To ensure we don't pick conflicting triples, we can maintain a set of used pairs
    used_pairs = set()
    
    for i, t in enumerate(all_triples):
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        conflict = False
        for p in pairs:
            if p in used_pairs:
                conflict = True
                break
        if not conflict:
            active_indices.append(i)
            for p in pairs:
                used_pairs.add(p)
                
    current_alpha = get_alpha(active_indices)
    best_alpha = current_alpha
    best_active_indices = list(active_indices)
    
    # Local Search
    # We try to swap triples or add triples.
    # Since we want to minimize alpha, adding triples is generally good.
    # But we are likely at a maximal packing already.
    # So we focus on swapping.
    
    # To speed up, we can cache the alpha calculation?
    # Or just run it. 2.4M ops is fast.
    
    for step in range(10000):
        if time.time() > deadline:
            break
            
        # Pick a random active triple to remove
        if not active_indices:
            break
        idx_to_remove = random.choice(active_indices)
        
        # Try to find a replacement triple that doesn't conflict with the rest
        # Or just try to add a new triple if we removed one?
        # Actually, we can just try random swaps.
        
        # Let's try adding a random triple that is not currently active and doesn't conflict
        # If we can add one, alpha might decrease.
        
        # First, try to add a triple
        added = False
        for _ in range(10): # Try a few random additions
             candidate_idx = random.choice(range(len(all_triples)))
             if candidate_idx in active_indices:
                 continue
             # Check conflict
             t = all_triples[candidate_idx]
             pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
             conflict = False
             for p in pairs:
                 if p in used_pairs:
                     conflict = True
                     break
             if not conflict:
                 # Add it
                 active_indices.append(candidate_idx)
                 for p in pairs:
                     used_pairs.add(p)
                 added = True
                 break
                 
        if added:
            new_alpha = get_alpha(active_indices)
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best_active_indices = list(active_indices)
                current_alpha = new_alpha
            else:
                # Revert
                # Find the index we just added
                # This is inefficient. Let's keep a stack or just rebuild used_pairs.
                # Rebuilding used_pairs is O(T).
                used_pairs = set()
                for i in active_indices:
                    t = all_triples[i]
                    used_pairs.add((t[0], t[1]))
                    used_pairs.add((t[0], t[2]))
                    used_pairs.add((t[1], t[2]))
                # Remove the last one
                last_idx = active_indices.pop()
                t = all_triples[last_idx]
                used_pairs.remove((t[0], t[1]))
                used_pairs.remove((t[0], t[2]))
                used_pairs.remove((t[1], t[2]))
                current_alpha = get_alpha(active_indices)
                
        # If we couldn't add, try a swap
        else:
             # Remove a random active triple
             remove_idx = random.choice(active_indices)
             t_rem = all_triples[remove_idx]
             rem_pairs = [(t_rem[0], t_rem[1]), (t_rem[0], t_rem[2]), (t_rem[1], t_rem[2])]
             
             # Tentatively remove
             for p in rem_pairs:
                 used_pairs.remove(p)
             active_indices.remove(remove_idx)
             
             # Try to add a different triple
             swapped = False
             for _ in range(10):
                 candidate_idx = random.choice(range(len(all_triples)))
                 if candidate_idx in active_indices:
                     continue
                 t_add = all_triples[candidate_idx]
                 add_pairs = [(t_add[0], t_add[1]), (t_add[0], t_add[2]), (t_add[1], t_add[2])]
                 conflict = False
                 for p in add_pairs:
                     if p in used_pairs:
                         conflict = True
                         break
                 if not conflict:
                     # Accept swap
                     active_indices.append(candidate_idx)
                     for p in add_pairs:
                         used_pairs.add(p)
                     swapped = True
                     break
             
             if swapped:
                 new_alpha = get_alpha(active_indices)
                 if new_alpha < best_alpha:
                     best_alpha = new_alpha
                     best_active_indices = list(active_indices)
                     current_alpha = new_alpha
                 else:
                     # Revert swap
                     # Remove candidate
                     last_idx = active_indices.pop()
                     t_add = all_triples[last_idx]
                     for p in [(t_add[0], t_add[1]), (t_add[0], t_add[2]), (t_add[1], t_add[2])]:
                         used_pairs.remove(p)
                     # Re-add removed
                     active_indices.append(remove_idx)
                     for p in rem_pairs:
                         used_pairs.add(p)
                     current_alpha = get_alpha(active_indices)
             else:
                 # Revert removal
                 active_indices.append(remove_idx)
                 for p in rem_pairs:
                     used_pairs.add(p)

    # Return the best solution found
    result = [list(all_triples[i]) for i in best_active_indices]
    return result
