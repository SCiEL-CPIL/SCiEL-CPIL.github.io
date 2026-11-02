<<<<<<< SEARCH
    best_set = []
    best_size = 0
    
    rng = np.random.default_rng(12345)
    
    # Strategy 1: Multiple greedy with different orderings
    # Key insight: vectors that are "more compatible" with many others should be picked first
    # Compute for each vector, how many others it's compatible with (dot <= 2)
    
    # compat_count[i] = number of j ≠ i such that M[i,j] <= 2
    # Exclude self: M[i,i] = 4 > 2, so we need to be careful
    compat = (M <= 2).astype(np.int32)
    # Subtract self-compatibility (M[i,i]=4, so compat[i,i]=0 already since 4>2)
    # Actually M[i,i] = 4, and 4 <= 2 is False, so compat[i,i] = 0. Good.
    compat_count = compat.sum(axis=1)
    
    # Sort by descending compatibility count (most compatible first)
    order_comp = np.argsort(-compat_count)
    
    orderings = [order_comp, np.arange(N)]
    # Add some random permutations
    for _ in range(50):
        orderings.append(rng.permutation(N))
    
    # Also try: sort by ascending index within groups of same type
    # Type 1: indices 0..21, Type 2: indices 22..N-1
    # For type 2, group by which 4 coordinates are used
    
    for oi, order in enumerate(orderings):
        if time.time() > deadline:
            break
            
        selected = []
        selected_mask = np.zeros(N, dtype=bool)
        
        for idx in order:
            if not selected:
                selected.append(idx)
                selected_mask[idx] = True
                continue
                
            # Check compatibility with all selected
            # M[idx, selected_mask] gives dot products with selected vectors
            # We need all to be <= 2
            dots = M[idx, selected_mask]
            if np.all(dots <= 2):
                selected.append(idx)
                selected_mask[idx] = True
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
            # If we hit 512, we're done
            if best_size >= 512:
                break
=======
    best_set = []
    best_size = 0
    
    rng = np.random.default_rng(12345)
    
    # Strategy 1: Multiple greedy with different orderings
    # Key insight: vectors that are "more compatible" with many others should be picked first
    # Compute for each vector, how many others it's compatible with (dot <= 2)
    
    # compat_count[i] = number of j ≠ i such that M[i,j] <= 2
    compat = (M <= 2).astype(np.int32)
    compat_count = compat.sum(axis=1)
    
    # Sort by descending compatibility count (most compatible first)
    order_comp = np.argsort(-compat_count)
    
    # Max-min diversity heuristic: pick first vector, then always pick the one 
    # that maximizes minimum distance to already-selected vectors
    def greedy_maxmin(start_idx, compat_mat, compat_cnt):
        """Greedy with max-min diversity after initial pick."""
        selected = [start_idx]
        selected_mask = np.zeros(N, dtype=bool)
        selected_mask[start_idx] = True
        
        # Precompute min distance to selected for each candidate
        min_dists = compat_mat.copy()  # will update
        
        for step in range(1, N):
            # Find candidate with max min-distance to selected
            # Only consider unselected candidates
            candidates_mask = ~selected_mask
            if not np.any(candidates_mask):
                break
            
            # For efficiency, track the minimum dot product violation count
            # Actually for max-min: we want to maximize the minimum "compatibility margin"
            # Use: among unselected, find one that has the largest minimum (2 - M[idx, sel])
            
            # Simple approach: for each unselected, compute min over selected of (2 - M[idx, sel])
            # This is expensive. Let's use a simpler heuristic:
            # Pick the unselected vector that is compatible with the most selected vectors
            # (i.e., has the fewest incompatibilities with current set)
            
            # Count incompatibilities with selected for each unselected
            # sub_M = M[np.ix_(np.where(candidates_mask)[0], np.where(selected_mask)[0])]
            # incomp = (sub_M > 2).sum(axis=1)
            # Pick argmin incomp, break ties by max compat_count
            
            # For speed, use vectorized approach
            sel_indices = np.where(selected_mask)[0]
            cand_indices = np.where(candidates_mask)[0]
            
            if len(cand_indices) == 0:
                break
            
            # Batch compute incompatibility counts
            # M[cand_indices][:, sel_indices] > 2, sum along axis 1
            sub = M[np.ix_(cand_indices, sel_indices)]
            incomp_counts = (sub > 2).sum(axis=1)
            
            # Pick the one with fewest incompatibilities
            min_incomp = np.min(incomp_counts)
            if min_incomp > 0:
                # No compatible candidate remains
                break
            
            # Among those with 0 incompatibilities, pick by highest compat_count
            zero_mask = incomp_counts == 0
            zero_cands = cand_indices[zero_mask]
            
            if len(zero_cands) == 0:
                break
            
            # Pick the one with highest original compat_count
            best_cand = zero_cands[np.argmax(compat_count[zero_cands])]
            
            selected.append(best_cand)
            selected_mask[best_cand] = True
        
        return selected
    
    orderings = [order_comp, np.arange(N)]
    # Add random permutations
    for _ in range(30):
        orderings.append(rng.permutation(N))
    
    # Run standard greedy on all orderings
    for oi, order in enumerate(orderings):
        if time.time() > deadline:
            break
            
        selected = []
        selected_mask = np.zeros(N, dtype=bool)
        
        for idx in order:
            if not selected:
                selected.append(idx)
                selected_mask[idx] = True
                continue
                
            dots = M[idx, selected_mask]
            if np.all(dots <= 2):
                selected.append(idx)
                selected_mask[idx] = True
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
            if best_size >= 512:
                break
    
    # Strategy 2: Max-min diversity greedy with different starting points
    # Try several different starting vectors
    start_candidates = list(order_comp[:10]) + list(rng.choice(N, size=20, replace=False))
    # Deduplicate
    start_candidates = list(dict.fromkeys(start_candidates))
    
    for start in start_candidates:
        if time.time() > deadline:
            break
        
        selected = greedy_maxmin(start, compat, compat_count)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
            if best_size >= 512:
                break
>>>>>>> REPLACE

