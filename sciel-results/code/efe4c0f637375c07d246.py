import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Generate Type 1 candidates: ±2e_i
    type1 = []
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            type1.append(v)
            
    # Generate Type 2 candidates: ±e_i ± e_j ± e_k ± e_l
    type2 = []
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            type2.append(v)
                
    type2 = np.array(type2, dtype=np.int32)
    N2 = len(type2)
    
    # Compute dot product matrix for Type 2
    # M[i,j] = type2[i] . type2[j]
    M = type2 @ type2.T
    # Conflict if M[i,j] > 2
    conflict = (M > 2).astype(np.uint8)
    np.fill_diagonal(conflict, 0)
    
    best_mask = np.zeros(N2, dtype=bool)
    best_count = 0
    rng = np.random.default_rng(42)
    
    # Strategy: Greedy + Local Search
    def greedy_select(order, conflict_mat, N):
        selected = np.zeros(N, dtype=bool)
        count = 0
        for idx in order:
            if not np.any(conflict_mat[idx] & selected):
                selected[idx] = True
                count += 1
        return selected, count

    # 1. Random Greedy Restarts
    n_restarts = 500
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        order = rng.permutation(N2)
        # Optional: sort by degree in the induced subgraph? 
        # For speed, just random permutation is often good enough for sparse graphs.
        # But let's try sorting by initial degree (number of conflicts)
        # This is O(N log N) per restart, might be slow for 500 restarts on 5280 nodes.
        # 5280 log 5280 ~ 5280 * 12 ~ 60k ops. 500 * 60k = 30M ops. Fast.
        
        # Calculate degrees once outside? No, degree in remaining graph changes.
        # Static degree is a good heuristic.
        # Let's use static degree for sorting.
        degrees = np.sum(conflict, axis=1)
        # To add randomness, we can shuffle ties or use a random permutation of indices with same degree?
        # Simpler: Just use random permutation for some, and sorted for others.
        
        if r % 2 == 0:
            # Random
            pass
        else:
            # Sorted by degree (lowest first)
            order = np.argsort(degrees, kind='stable')
            # Add noise
            # order = order + rng.integers(-10, 10, size=len(order)) # Risky
            
        selected, count = greedy_select(order, conflict, N2)
        
        if count > best_count:
            best_count = count
            best_mask = selected.copy()
            
    # 2. Local Search on the best solution found
    if time.time() < deadline - 10:
        # Phase 1: Try to add any compatible unselected vectors
        unselected = np.where(~best_mask)[0]
        for idx in unselected:
            if time.time() > deadline - 5:
                break
            if not np.any(conflict[idx] & best_mask):
                best_mask[idx] = True
                best_count += 1
                
        # Phase 2: Swap-based Local Search
        # Try to swap a selected vertex with an unselected one if it allows adding more later?
        # Simpler: 1-move local search.
        # If we remove u, can we add more than 1?
        # This is expensive.
        # Let's do "Greedy Re-add":
        # Remove a random small set, re-add greedily.
        
        for ls_iter in range(100):
            if time.time() > deadline - 5:
                break
            
            # Pick 3 random selected vertices to remove
            selected_indices = np.where(best_mask)[0]
            if len(selected_indices) < 3:
                break
                
            n_remove = 3
            remove_indices = rng.choice(selected_indices, size=n_remove, replace=False)
            
            test_mask = best_mask.copy()
            for r_idx in remove_indices:
                test_mask[r_idx] = False
            test_count = best_count - n_remove
            
            # Greedily add back from unselected
            unselected_test = np.where(~test_mask)[0]
            # Sort unselected by conflict count with test_mask
            # This is O(N^2) if done naively. 
            # conflict[u] & test_mask is a vector AND and sum.
            # Vectorized: 
            # conf_counts = np.array([np.sum(conflict[u] & test_mask) for u in unselected_test])
            # This loop is slow in Python.
            # Use matrix multiplication?
            # conflict @ test_mask gives number of conflicts for each vertex.
            conf_counts = conflict @ test_mask
            # Filter to unselected
            conf_counts_unsel = conf_counts[unselected_test]
            order_add = unselected_test[np.argsort(conf_counts_unsel)]
            
            for add_idx in order_add:
                if time.time() > deadline - 5:
                    break
                if not np.any(conflict[add_idx] & test_mask):
                    test_mask[add_idx] = True
                    test_count += 1
                    
            if test_count > best_count:
                best_mask = test_mask
                best_count = test_count
            else:
                # If no improvement, maybe try a different removal set in next iter
                pass
                
    # Construct final array
    # Type 1 vectors are always compatible with Type 2
    # And mutually compatible.
    type1_arr = np.array(type1, dtype=np.int64)
    type2_selected = type2[best_mask].astype(np.int64)
    
    final_vectors = np.vstack([type1_arr, type2_selected])
    
    return final_vectors
