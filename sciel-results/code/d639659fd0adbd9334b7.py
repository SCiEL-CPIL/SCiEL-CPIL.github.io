<<<<<<< SEARCH
def _greedy_construct(v, k, t, deadline, rng):
    all_blocks, block_subsets, blocks_covering_subset, n_blocks, n_subsets = _prepare(v, k, t)
    
    uncovered = np.ones(n_subsets, dtype=np.int8)
    counts = np.zeros(n_subsets, dtype=np.int32)
    selected = []
    selected_set = set()
    
    # Build a sparse incidence matrix: block_subsets as a list of arrays
    block_subset_arrays = [np.array(subs, dtype=np.int32) for subs in block_subsets]
    
    while uncovered.any() and time.time() < deadline:
        # Vectorized scoring: for each unselected block, count uncovered subsets it covers
        best_block = -1
        best_score = -1
        
        # Process in chunks to avoid huge memory usage
        chunk_size = 5000
        for start in range(0, n_blocks, chunk_size):
            end = min(start + chunk_size, n_blocks)
            for i in range(start, end):
                if i in selected_set:
                    continue
                subs = block_subset_arrays[i]
                score = uncovered[subs].sum()
                if score > best_score:
                    best_score = score
                    best_block = i
        
        if best_block == -1 or best_score <= 0:
            break
            
        selected.append(best_block)
        selected_set.add(best_block)
        
        # Update uncovered and counts
        subs = block_subset_arrays[best_block]
        counts[subs] += 1
        uncovered[subs] = 0
        
    return selected, counts.tolist(), selected_set
=======
def _greedy_construct(v, k, t, deadline, rng):
    all_blocks, block_subsets, blocks_covering_subset, n_blocks, n_subsets = _prepare(v, k, t)
    
    uncovered = np.ones(n_subsets, dtype=np.int8)
    counts = np.zeros(n_subsets, dtype=np.int32)
    selected = []
    selected_set = set()
    
    # Build a sparse incidence matrix: block_subsets as a list of arrays
    block_subset_arrays = [np.array(subs, dtype=np.int32) for subs in block_subsets]
    
    # Adaptive weights for points - start uniform, learn from uncovered subsets
    point_weights = np.ones(v, dtype=np.float64)
    
    # Precompute: for each block, which points it contains (as a set for fast lookup)
    block_points = [set(b) for b in all_blocks]
    
    # For efficiency, precompute for each subset which points it contains
    subset_points = [list(s) for s in itertools.combinations(range(v), t)]
    
    while uncovered.any() and time.time() < deadline:
        # Compute weighted score for each block
        # Score = sum over uncovered subsets in block of (product of weights of points in subset)
        # This biases toward blocks covering "high-need" regions
        
        best_block = -1
        best_score = -1.0
        
        # Process in chunks to avoid huge memory usage
        chunk_size = 2000
        for start in range(0, n_blocks, chunk_size):
            end = min(start + chunk_size, n_blocks)
            for i in range(start, end):
                if i in selected_set:
                    continue
                subs = block_subset_arrays[i]
                # For each uncovered subset in this block, compute weight product
                # Use the point weights to compute a bias factor
                block_pts = block_points[i]
                score = 0.0
                for s_idx in subs:
                    if uncovered[s_idx]:
                        # Compute product of weights for points in this subset
                        s_pts = subset_points[s_idx]
                        w_prod = 1.0
                        for p in s_pts:
                            w_prod *= point_weights[p]
                        score += w_prod
                if score > best_score:
                    best_score = score
                    best_block = i
        
        if best_block == -1 or best_score <= 0:
            break
            
        selected.append(best_block)
        selected_set.add(best_block)
        
        # Update uncovered and counts
        subs = block_subset_arrays[best_block]
        counts[subs] += 1
        uncovered[subs] = 0
        
        # Update point weights: increase weights for points in still-uncovered subsets
        # Points that appear in many uncovered subsets should get higher weight
        # This is a simple heuristic: for each uncovered subset, add to each point's weight
        # But this is O(n_subsets * t) per iteration which is expensive.
        # Instead, periodically update weights based on coverage deficit
        
        # Periodic weight update (every 10 blocks or so)
        if len(selected) % 10 == 0 and time.time() < deadline:
            # For each point, compute how many uncovered subsets contain it
            point_deficit = np.zeros(v, dtype=np.float64)
            for s_idx in range(n_subsets):
                if uncovered[s_idx]:
                    for p in subset_points[s_idx]:
                        point_deficit[p] += 1.0
            # Update weights: multiply by (1 + deficit / max_deficit)
            max_deficit = np.max(point_deficit) if np.max(point_deficit) > 0 else 1.0
            point_weights *= (1.0 + point_deficit / max_deficit)
            # Normalize to prevent overflow
            point_weights /= np.max(point_weights)
        
    return selected, counts.tolist(), selected_set
>>>>>>> REPLACE

