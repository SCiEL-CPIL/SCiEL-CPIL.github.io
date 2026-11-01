"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    # Strategy:
    # Use vectors with norm squared = 4 (Norm = 2).
    # The constraint is min pairwise dist >= max norm.
    # If all vectors have norm 2, we need dist^2 >= 4.
    # dist^2 = ||u||^2 + ||v||^2 - 2 u.v = 8 - 2 u.v >= 4 => u.v <= 2.
    
    # Type A: Weight 1 vectors in {-2, 0, 2}^11. Norm^2 = 4.
    # Type C: Weight 4 vectors in {-1, 0, 1}^11. Norm^2 = 4.
    
    # Generate W4 vectors
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate W1 vectors
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
    
    # Compute conflict matrix for W4: conflict if u.v > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Also check W1 vs W4 compatibility
    # W1 vectors have entries 0 or ±2, W4 have entries 0 or ±1
    # dot product max is 2*1 = 2 (if they share a position with same sign)
    # So all W1 are compatible with all W4 (dot <= 2 always)
    
    # W1 vs W1: same position different sign: dot = -4 <= 2. Same pos same sign: dot=4 > 2!
    # So we can't include both +e_i*2 and -e_i*2... wait, dot = 4 > 2, conflict!
    # Actually wait: if u = 2*e_i and v = 2*e_i, dot = 4 > 2, CONFLICT.
    # If u = 2*e_i and v = -2*e_i, dot = -4 <= 2, OK.
    # If u = 2*e_i and v = 2*e_j (i≠j), dot = 0 <= 2, OK.
    # So among W1, we can pick at most one from each {+2e_i, -2e_i} pair.
    # We can pick all 11 with consistent signs, or mix.
    # Actually we can pick any subset where no two are identical.
    # Since +2e_i and -2e_i are different vectors, and their dot is -4, they're compatible.
    # But +2e_i and +2e_i is the same vector, so we only have one of each.
    # The 22 W1 vectors: for each i, we have +2e_i and -2e_i.
    # +2e_i dot +2e_i = 4 (same vector, not applicable since we pick distinct)
    # +2e_i dot -2e_i = -4 <= 2, compatible
    # +2e_i dot +2e_j = 0 for i≠j, compatible
    # So ALL 22 W1 vectors are mutually compatible!
    
    # Now do multiple random restarts of greedy min-degree MIS on W4
    best_selected = None
    best_count = 0
    
    n_restarts = 50
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 15:
            break
            
        alive = np.ones(N4, dtype=bool)
        selected_indices = []
        degrees = conflicts.sum(axis=1).astype(np.int32)
        
        while np.any(alive):
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            
            # Random tie-breaking
            if len(cands) > 1:
                chosen = cands[rng.integers(len(cands))]
            else:
                chosen = cands[0]
            
            selected_indices.append(chosen)
            
            neighbors = np.where(conflicts[chosen])[0]
            to_remove = np.unique(np.concatenate(([chosen], neighbors)))
            
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
        
        count = len(selected_indices)
        if count > best_count:
            best_count = count
            best_selected = selected_indices[:]
        
        # Try local improvement: try to add back some removed vertices
        if time.time() > deadline - 10:
            break
            
        # Local search: try swapping selected vertices with unselected ones
        selected_set = set(selected_indices)
        improved = True
        max_improve_iters = 500
        improve_iter = 0
        
        while improved and improve_iter < max_improve_iters:
            improved = False
            improve_iter += 1
            
            if time.time() > deadline - 10:
                break
            
            # For each selected vertex, try replacing with an unselected vertex
            # that has fewer conflicts with the remaining selected set
            unselected = [i for i in range(N4) if i not in selected_set]
            
            # Sample some swaps for efficiency
            if len(unselected) > 200:
                swap_candidates = rng.choice(unselected, size=200, replace=False)
            else:
                swap_candidates = unselected
            
            best_swap = None
            best_score = len(selected_indices)
            
            for new_v in swap_candidates:
                # Check if new_v is compatible with all selected except potentially one
                # We'll try: remove one selected, add new_v
                # Score = number of conflicts of new_v with (selected - {removed})
                
                # Compute conflicts of new_v with all selected
                sel_arr = np.array(list(selected_set))
                if len(sel_arr) == 0:
                    continue
                new_conflicts = conflicts[new_v, sel_arr]
                n_conf = new_conflicts.sum()
                
                if n_conf == 0:
                    # new_v is compatible with all selected, just add it
                    selected_set.add(new_v)
                    selected_indices.append(new_v)
                    improved = True
                    break
                elif n_conf == 1:
                    # Find which selected vertex conflicts
                    conflict_idx = np.where(new_conflicts)[0][0]
                    remove_v = sel_arr[conflict_idx]
                    # Check: after removing remove_v, is new_v compatible with rest?
                    # Yes, by construction (only 1 conflict)
                    # But we also need to check that remove_v was only conflicting with new_v
                    # among the remaining... actually we just need the final set to be valid
                    selected_set.remove(remove_v)
                    selected_set.add(new_v)
                    selected_indices = list(selected_set)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try adding a vertex that's compatible with all selected
            for new_v in swap_candidates:
                if new_v in selected_set:
                    continue
                sel_arr = np.array(list(selected_set))
                if len(sel_arr) == 0:
                    selected_set.add(new_v)
                    selected_indices.append(new_v)
                    improved = True
                    break
                if not conflicts[new_v, sel_arr].any():
                    selected_set.add(new_v)
                    selected_indices.append(new_v)
                    improved = True
                    break
        
        count = len(selected_indices)
        if count > best_count:
            best_count = count
            best_selected = selected_indices[:]
    
    if best_selected is None:
        best_selected = []
    
    result_v4 = vectors_v4[best_selected]
    
    # Combine with W1
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
