import numpy as np
import time
import itertools

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Generate all integer vectors in R^11 with norm 4 (squared norm 16)
    # We focus on the subset with exactly four non-zero entries of magnitude 2.
    # v = (2, 2, 2, 2, 0, ..., 0) with signs.
    
    vectors = []
    # Indices for the 4 non-zero positions
    for indices in itertools.combinations(range(D), 4):
        # Signs for the 4 non-zero positions
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(indices, signs):
                v[idx] = 2 * s
            vectors.append(v)
    
    V = np.array(vectors, dtype=np.int64)
    n_vectors = len(V)
    
    # Compute pairwise dot products
    # We need to find pairs where dot_product > 8
    # dot(v, w) = sum(v_i * w_i)
    # Since v, w are sparse (4 non-zeros), we can compute this efficiently.
    # But for 5280 vectors, matrix multiplication is fine.
    
    # To save memory and time, we can compute in batches or just use V @ V.T
    # 5280 x 5280 matrix is ~28 million elements. 28M * 8 bytes = 224 MB. Acceptable.
    
    dot_products = V @ V.T
    
    # Conflict condition: dot_product > 8
    # Note: v . v = 16. We ignore self-loops.
    
    conflict_mask = dot_products > 8
    np.fill_diagonal(conflict_mask, False)
    
    # Find a large independent set
    # Greedy approach:
    # 1. Compute degrees (number of conflicts)
    degrees = conflict_mask.sum(axis=1)
    
    # Sort indices by degree (ascending)
    order = np.argsort(degrees)
    
    selected = []
    selected_set = set()
    
    # Greedy selection
    for idx in order:
        # Check if idx conflicts with any selected node
        # selected_set contains indices
        # We can check conflict_mask[idx, selected_set]
        # But iterating over selected_set is slow.
        # Better: maintain a 'blocked' array.
        
        pass

    # Better Greedy:
    blocked = np.zeros(n_vectors, dtype=bool)
    selected_indices = []
    
    # We need to check if any neighbor is already selected.
    # If we select idx, we must block all its neighbors.
    
    # Let's use a priority queue or just iterate sorted by degree.
    # If we pick a low-degree node, we block few others.
    
    for idx in order:
        if not blocked[idx]:
            selected_indices.append(idx)
            # Block all neighbors of idx
            neighbors = np.where(conflict_mask[idx])[0]
            blocked[neighbors] = True
            blocked[idx] = True # Mark as processed/selected
            
    current_best = np.array(V[selected_indices], dtype=np.int64)
    
    # Local Search / Hill Climbing to improve
    # Try to swap one selected vector with one unselected vector if it increases size?
    # No, we want to maximize size.
    # If we can add a vector without removing any, we do it.
    # But greedy usually gets stuck.
    
    # Let's try a simple local search:
    # Randomly pick a selected vector, remove it, and see if we can add more.
    # Or simply: try to add any unselected vector that doesn't conflict.
    
    # Actually, the greedy above is quite strong for sparse graphs.
    # Let's add a pass to see if we can add more.
    
    # Re-check for addable nodes
    # This is O(N^2) per pass, but N is small.
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Find unselected nodes
        unselected = [i for i in range(n_vectors) if i not in set(selected_indices)]
        # This is slow. Use boolean array.
        is_selected = np.zeros(n_vectors, dtype=bool)
        is_selected[selected_indices] = True
        
        # For each unselected node, check if it conflicts with any selected node.
        # conflict_mask[unselected, selected_indices].any(axis=1)
        
        if len(unselected) > 0:
            unselected_arr = np.array(unselected)
            conflicts_with_selected = conflict_mask[unselected_arr][:, is_selected].any(axis=1)
            
            # Find indices where conflicts_with_selected is False
            addable = unselected_arr[~conflicts_with_selected]
            
            if len(addable) > 0:
                # Add all addable nodes?
                # Adding one might block others.
                # Just add the first one and repeat.
                # Or add all if they don't conflict with each other?
                # Simpler: add one, update is_selected, loop.
                
                # To be safe and fast, just add the first one found.
                idx_to_add = addable[0]
                selected_indices.append(idx_to_add)
                is_selected[idx_to_add] = True
                changed = True
                
                # Update blocked logic? 
                # The next iteration of the while loop will re-evaluate.
                
    # Final result
    result = V[selected_indices]
    
    # Verify constraints (optional but good for debugging)
    # if len(result) > 1:
    #     dists = np.sqrt(np.sum((result[:, np.newaxis, :] - result[np.newaxis, :, :])**2, axis=2))
    #     np.fill_diagonal(dists, np.inf)
    #     min_dist = np.min(dists)
    #     max_norm = np.max(np.linalg.norm(result, axis=1))
    #     assert min_dist >= max_norm - 1e-9, f"Invalid: {min_dist} < {max_norm}"
        
    return result
# EVOLVE-BLOCK-END
