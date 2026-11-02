"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # 1. Generate Candidate Pools
    # We will use a mix of weight-2 and weight-4 vectors.
    # Weight 2: 2 non-zero coords, values +/-1. Norm^2 = 2.
    # Weight 4: 4 non-zero coords, values +/-1. Norm^2 = 4.
    
    # The condition is min pairwise distance >= max norm.
    # If we mix norms, max_norm will be 2 (norm of weight-4 vectors).
    # Condition: ||v_i - v_j|| >= 2  => ||v_i - v_j||^2 >= 4.
    # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2 v_i.v_j.
    
    # Case 1: v_i is w2 (norm^2=2), v_j is w4 (norm^2=4).
    # 2 + 4 - 2 v_i.v_j >= 4  =>  6 - 2 v_i.v_j >= 4  =>  2 >= 2 v_i.v_j  =>  v_i.v_j <= 1.
    
    # Case 2: v_i is w2, v_j is w2.
    # 2 + 2 - 2 v_i.v_j >= 4  =>  4 - 2 v_i.v_j >= 4  =>  v_i.v_j <= 0.
    
    # Case 3: v_i is w4, v_j is w4.
    # 4 + 4 - 2 v_i.v_j >= 4  =>  8 - 2 v_i.v_j >= 4  =>  v_i.v_j <= 2.
    
    # So we have different compatibility constraints depending on the types!
    # This makes a unified "conflict graph" approach complex if we just use dot products.
    # However, we can still define a conflict:
    # Conflict(i, j) if:
    #   (type_i == 2 and type_j == 2) and dot(i,j) > 0
    #   (type_i == 2 and type_j == 4) and dot(i,j) > 1
    #   (type_i == 4 and type_j == 2) and dot(i,j) > 1
    #   (type_i == 4 and type_j == 4) and dot(i,j) > 2
    
    # Let's generate all w2 and w4 vectors.
    candidates = []
    types = [] # 0 for w2, 1 for w4
    
    # Generate Weight 2 vectors
    # C(11,2) * 2^2 = 55 * 4 = 220 vectors
    for comb in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = np.zeros(D, dtype=np.int64)
            v[comb[0]] = signs[0]
            v[comb[1]] = signs[1]
            candidates.append(v)
            types.append(0)
            
    # Generate Weight 4 vectors
    # C(11,4) * 2^4 = 330 * 16 = 5280 vectors
    for comb in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            candidates.append(v)
            types.append(1)
            
    candidates = np.array(candidates)
    types = np.array(types)
    n_cand = len(candidates)
    
    # Precompute Gram matrix
    G = candidates @ candidates.T
    
    # Precompute conflict matrix based on types
    # We need to apply the threshold based on the pair of types.
    # Let's create a mask for each type pair.
    # T_ii: both type 0 (w2). Threshold dot > 0.
    # T_ij: one type 0, one type 1 (w2, w4). Threshold dot > 1.
    # T_jj: both type 1 (w4). Threshold dot > 2.
    
    # To do this efficiently without loops over all pairs (which is 5500x5500 ~ 30M ops, manageable in numpy):
    # We can compute the dot products and then use broadcasting to apply thresholds.
    
    # Create type matrix (n_cand, n_cand)
    type_mat = np.add.outer(types, types) # 0:0+0=0, 0:1+0=1, 1:1+1=2
    
    # Threshold matrix
    # Where type_mat == 0 (w2-w2): conflict if G > 0
    # Where type_mat == 1 (w2-w4): conflict if G > 1
    # Where type_mat == 2 (w4-w4): conflict if G > 2
    
    conflict_mat = np.zeros((n_cand, n_cand), dtype=bool)
    
    # Apply conditions
    # Note: G is symmetric.
    mask_0 = (type_mat == 0)
    mask_1 = (type_mat == 1)
    mask_2 = (type_mat == 2)
    
    conflict_mat[mask_0] = G[mask_0] > 0
    conflict_mat[mask_1] = G[mask_1] > 1
    conflict_mat[mask_2] = G[mask_2] > 2
    
    np.fill_diagonal(conflict_mat, False)
    
    # 2. Greedy Initialization
    # We want to maximize the number of selected vectors.
    # A good heuristic is to pick vectors with the fewest conflicts first (least restrictive).
    # Or, since we want to mix types, we can just pick randomly but ensure validity.
    # Let's try a "Least Conflicts First" greedy.
    
    # Count conflicts for each node
    conflict_counts = np.sum(conflict_mat, axis=1)
    
    # Order candidates by conflict count (ascending)
    order = np.argsort(conflict_counts)
    
    selected = []
    selected_set = set()
    
    for idx in order:
        # Check if idx conflicts with any already selected
        # We can check G[idx, selected] against thresholds? 
        # No, we precomputed conflict_mat. 
        # Check if any selected node is in conflict with idx.
        # Since selected is a list, we can just check conflict_mat[idx, selected]
        # For speed, we can maintain a "blocked" array? 
        # Actually, checking conflict_mat[idx, selected] is O(|selected|).
        # Total O(N^2) worst case, but usually much less as we stop early or select few.
        
        if not selected:
            selected.append(idx)
            selected_set.add(idx)
        else:
            # Check against all selected
            # If any conflict, skip.
            # conflict_mat[idx, selected] is a boolean array.
            if not np.any(conflict_mat[idx, selected]):
                selected.append(idx)
                selected_set.add(idx)
                
    # 3. Local Search (1-Swap and 2-Swap)
    # We try to improve the solution by swapping selected vectors with unselected ones,
    # or adding new ones if possible (though greedy should have added all possible non-conflicting ones? 
    # No, greedy is order-dependent. A later vector might fit if we remove an earlier one).
    
    # Let's implement a simple hill-climbing step.
    # We will iterate through unselected vectors and see if swapping with a selected one improves the count 
    # (actually, swapping keeps count same, but might allow adding MORE later? 
    # Better: try to ADD a vector by removing MINIMUM necessary conflicts? 
    # Or simpler: Random Restart Greedy.
    
    # Given time constraints, let's try a few random restarts of the greedy with different orderings.
    best_selected = selected
    best_count = len(selected)
    
    # We have ~140s. 
    # Greedy is fast. Let's run multiple restarts with random permutations.
    rng = np.random.default_rng(42)
    
    # How many restarts? 
    # One greedy pass is O(N^2) ~ 30M ops. In Python/NumPy, this is fast (<1s).
    # We can do 50-100 restarts.
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        # Random permutation of candidates
        perm = rng.permutation(n_cand)
        
        curr_selected = []
        curr_selected_set = set()
        
        for idx in perm:
            if not curr_selected:
                curr_selected.append(idx)
                curr_selected_set.add(idx)
            else:
                if not np.any(conflict_mat[idx, curr_selected]):
                    curr_selected.append(idx)
                    curr_selected_set.add(idx)
                    
        curr_count = len(curr_selected)
        if curr_count > best_count:
            best_count = curr_count
            best_selected = curr_selected
            best_selected_set = curr_selected_set
            
            # Optional: Local search on best found
            # Try to swap in unselected vectors if they allow adding more?
            # This is complex. Let's stick to restarts for now as they are robust.
            
    # 4. Final Local Search on the best solution
    # Try to swap selected vectors with unselected ones to see if we can unlock more additions?
    # Actually, a simpler local search: 
    # For each unselected vector u, count how many selected vectors it conflicts with.
    # If it conflicts with 0, it should have been added (greedy failed due to order? No, if 0 conflicts, it's compatible with ALL selected).
    # Wait, if u has 0 conflicts with current selected, it MUST be compatible. 
    # Why wasn't it added? 
    # In the random restart greedy, if u was processed BEFORE some selected vectors were added, 
    # it might have been skipped if it conflicted with those earlier ones? 
    # No, if it's compatible with the FINAL set, it might not have been compatible with the PARTIAL set at the time it was considered?
    # Actually, if u is compatible with the final set S, it is compatible with any subset of S.
    # So if u was considered at time t, and S_t (selected so far) was a subset of S_final, 
    # and u was compatible with S_final, it was compatible with S_t.
    # So u WOULD have been added.
    # Therefore, any vector compatible with the final set MUST be in the final set.
    # This means the greedy set is "maximal" (no single vector can be added without removing another).
    
    # So to improve, we MUST swap.
    # Heuristic: 
    # Find a selected vector s that conflicts with many unselected vectors (it's "blocking" many).
    # Find an unselected vector u that conflicts with few selected vectors (it's "almost" compatible).
    # If we remove s, can we add u AND some others?
    
    # Let's try a simple 1-swap: 
    # For each unselected u, let C_u be the set of selected vectors it conflicts with.
    # If we remove ALL of C_u, we can add u.
    # Net change in count = 1 - |C_u|.
    # We only want to do this if |C_u| == 0 (impossible as shown above) or if we can add MORE than |C_u| vectors.
    # This is a set cover problem.
    
    # Given time, let's just return the best greedy result. 
    # The mixing of weights is the key structural change.
    
    if best_selected:
        final_vecs = candidates[best_selected]
        return final_vecs
    else:
        return np.array([np.ones(D, dtype=np.int64)], dtype=np.int64)

# EVOLVE-BLOCK-END
