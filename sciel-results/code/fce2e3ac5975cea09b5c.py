"""Optimized construction of large 4-uniform sunflower-free family."""

def construct():
    from itertools import combinations
    import time
    import numpy as np
    
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    # Generate all 4-subsets of [1,16]
    all_sets = list(combinations(range(1, 17), 4))
    n_total = len(all_sets)  # 1820
    
    # Convert to bitmasks
    def to_mask(s):
        m = 0
        for x in s:
            m |= (1 << (x - 1))
        return m
    
    masks = np.array([to_mask(s) for s in all_sets], dtype=np.int64)
    
    def check_conflict_fast(c_mask, chosen_arr, pairwise_inter):
        """Check if c_mask conflicts with any pair in chosen_arr.
        
        Uses precomputed pairwise intersection matrix for speed.
        """
        n = len(chosen_arr)
        if n < 2:
            return False
        
        # I_i = chosen[i] & c_mask
        I = chosen_arr & c_mask  # shape (n,)
        
        # Group indices by their intersection value with c
        # For efficiency, use dictionary
        from collections import defaultdict
        groups = defaultdict(list)
        for i in range(n):
            groups[int(I[i])].append(i)
        
        # For each group with >= 2 members, check pairwise using precomputed intersections
        for t, idxs in groups.items():
            if len(idxs) < 2:
                continue
            # Check all pairs in this group using precomputed pairwise_inter
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    i, j = idxs[a], idxs[b]
                    if pairwise_inter[i, j] == t:
                        return True
        return False
    
    def greedy_build(order, masks, deadline, start_time=None):
        """Greedy build following given order with incremental pairwise matrix."""
        if start_time is None:
            start_time = time.time()
        
        chosen_indices = []
        chosen_masks_list = []
        # Pairwise intersection matrix (n x n), built incrementally
        pairwise = None
        
        for idx in order:
            if time.time() > deadline:
                break
            cm = masks[idx]
            n_chosen = len(chosen_masks_list)
            
            if n_chosen < 2:
                chosen_indices.append(idx)
                chosen_masks_list.append(cm)
                if n_chosen == 1:
                    pairwise = np.zeros((2, 2), dtype=np.int64)
                    pairwise[0, 1] = chosen_masks_list[0] & chosen_masks_list[1]
                    pairwise[1, 0] = pairwise[0, 1]
                continue
            
            chosen_arr = np.array(chosen_masks_list, dtype=np.int64)
            
            if not check_conflict_fast(cm, chosen_arr, pairwise):
                # Add to family
                new_n = n_chosen + 1
                new_pairwise = np.zeros((new_n, new_n), dtype=np.int64)
                if pairwise is not None:
                    new_pairwise[:n_chosen, :n_chosen] = pairwise
                
                # Compute intersections of new set with all existing
                new_inter = chosen_arr & cm  # shape (n_chosen,)
                new_pairwise[:n_chosen, n_chosen] = new_inter
                new_pairwise[n_chosen, :n_chosen] = new_inter
                new_pairwise[n_chosen, n_chosen] = 0
                
                chosen_indices.append(idx)
                chosen_masks_list.append(cm)
                pairwise = new_pairwise
        
        return chosen_indices, chosen_masks_list, pairwise
    
    def local_improve(chosen_indices, chosen_masks_list, pairwise, masks, deadline, n_total):
        """Try to improve by removing one element and adding multiple new ones."""
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            n_chosen = len(chosen_indices)
            if n_chosen < 2:
                break
            chosen_set = set(chosen_indices)
            
            # Try removing each chosen element
            for remove_pos in range(n_chosen):
                if time.time() > deadline - 3:
                    break
                
                # Remove element at remove_pos
                remaining_indices = chosen_indices[:remove_pos] + chosen_indices[remove_pos+1:]
                remaining_masks = [chosen_masks_list[i] for i in range(n_chosen) if i != remove_pos]
                
                if len(remaining_masks) < 2:
                    continue
                
                remaining_arr = np.array(remaining_masks, dtype=np.int64)
                n_rem = len(remaining_masks)
                
                # Rebuild pairwise matrix for remaining
                rem_pairwise = np.zeros((n_rem, n_rem), dtype=np.int64)
                for i in range(n_rem):
                    for j in range(i+1, n_rem):
                        rem_pairwise[i, j] = remaining_masks[i] & remaining_masks[j]
                        rem_pairwise[j, i] = rem_pairwise[i, j]
                
                # Try to add elements not in chosen
                added = []
                added_masks = []
                for idx in range(n_total):
                    if idx in chosen_set:
                        continue
                    if time.time() > deadline - 4:
                        break
                    cm = masks[idx]
                    
                    if len(added_masks) < 2:
                        # Few existing, just add
                        added.append(idx)
                        added_masks.append(cm)
                        continue
                    
                    # Check conflict against remaining + added
                    all_existing = remaining_arr
                    if added_masks:
                        all_existing = np.concatenate([remaining_arr, np.array(added_masks, dtype=np.int64)])
                    
                    # Build full pairwise for checking
                    n_all = len(all_existing)
                    full_pairwise = np.zeros((n_all, n_all), dtype=np.int64)
                    if n_rem >= 2:
                        full_pairwise[:n_rem, :n_rem] = rem_pairwise
                    
                    # Fill in added-added and remaining-added intersections
                    for i in range(n_all):
                        for j in range(i+1, n_all):
                            if i >= n_rem or j >= n_rem:
                                full_pairwise[i, j] = all_existing[i] & all_existing[j]
                                full_pairwise[j, i] = full_pairwise[i, j]
                    
                    if not check_conflict_fast(cm, all_existing, full_pairwise):
                        added.append(idx)
                        added_masks.append(cm)
                
                # If we added more than we removed (1), it's an improvement
                if len(added) > 1:
                    new_indices = remaining_indices + added
                    new_masks = remaining_masks + added_masks
                    chosen_indices = new_indices
                    chosen_masks_list = new_masks
                    
                    # Rebuild pairwise
                    n_new = len(new_masks)
                    pairwise = np.zeros((n_new, n_new), dtype=np.int64)
                    for i in range(n_new):
                        for j in range(i+1, n_new):
                            pairwise[i, j] = new_masks[i] & new_masks[j]
                            pairwise[j, i] = pairwise[i, j]
                    
                    improved = True
                    break
        
        return chosen_indices, chosen_masks_list, pairwise
    
    # First try sequential order
    order_seq = list(range(n_total))
    chosen_indices, chosen_masks_list, pairwise = greedy_build(order_seq, masks, deadline)
    
    best_size = len(chosen_indices)
    best_indices = list(chosen_indices)
    best_masks = list(chosen_masks_list)
    best_pairwise = pairwise
    
    # Local improvement on sequential result
    if time.time() < deadline - 10:
        ci, cm, pw = local_improve(chosen_indices, chosen_masks_list, pairwise, masks, deadline, n_total)
        if len(ci) > best_size:
            best_size = len(ci)
            best_indices = list(ci)
            best_masks = list(cm)
            best_pairwise = pw
    
    # Try random restarts if time permits
    rng = np.random.default_rng(42)
    restarts = 0
    while time.time() < deadline - 15 and restarts < 10:
        order = rng.permutation(n_total)
        chosen_r, masks_r, pw_r = greedy_build(order, masks, deadline)
        
        # Local improvement
        if time.time() < deadline - 10 and len(chosen_r) >= best_size - 5:
            ci, cm, pw = local_improve(chosen_r, masks_r, pw_r, masks, deadline, n_total)
            if len(ci) > len(chosen_r):
                chosen_r, masks_r, pw_r = ci, cm, pw
        
        restarts += 1
        if len(chosen_r) > best_size:
            best_size = len(chosen_r)
            best_indices = list(chosen_r)
            best_masks = list(masks_r)
            best_pairwise = pw_r
    
    # Try degree-based ordering: order by number of conflicts with early sets
    if time.time() < deadline - 20:
        # Compute a heuristic: for each set, count how many other sets it conflicts with
        # (forms sunflower with some pair)
        # This is expensive, so use a simpler heuristic: order by sum of element values
        order_deg = sorted(range(n_total), key=lambda i: sum(all_sets[i]))
        chosen_d, masks_d, pw_d = greedy_build(order_deg, masks, deadline)
        
        if time.time() < deadline - 10 and len(chosen_d) >= best_size - 5:
            ci, cm, pw = local_improve(chosen_d, masks_d, pw_d, masks, deadline, n_total)
            if len(ci) > len(chosen_d):
                chosen_d, masks_d, pw_d = ci, cm, pw
        
        if len(chosen_d) > best_size:
            best_size = len(chosen_d)
            best_indices = list(chosen_d)
            best_masks = list(masks_d)
            best_pairwise = pw_d
    
    return [sorted(all_sets[i]) for i in best_indices]
