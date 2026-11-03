"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # 1. Generate Candidate Pools
    
    # Type A: +/- 2e_i (22 vectors)
    # These are compatible with ALL Type B vectors and ALL other Type A vectors.
    type_a_list = []
    for i in range(D):
        for sign in (1, -1):
            v = [0] * D
            v[i] = sign * 2
            type_a_list.append(v)
    type_a = np.array(type_a_list, dtype=np.int8)
    
    # Type B: +/- e_i +/- e_j +/- e_k +/- e_l (5280 vectors)
    type_b_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            type_b_list.append(v)
    type_b = np.array(type_b_list, dtype=np.int8)
    n_b = len(type_b)
    
    # 2. Precompute Conflict Matrix for Type B
    # Conflict if dot > 2
    c_b = type_b.astype(np.int32)
    dot_b = c_b @ c_b.T
    conflict_b = (dot_b > 2).astype(np.uint8)
    np.fill_diagonal(conflict_b, 0)
    
    best_count_b = 0
    best_selected_b = np.zeros(n_b, dtype=np.uint8)
    
    # 3. Multi-Start Search on Type B
    rng = np.random.default_rng(42)
    
    def greedy_construct_b():
        """
        Greedy selection using minimum-degree heuristic on the remaining graph of Type B.
        """
        selected = np.zeros(n_b, dtype=np.uint8)
        remaining = np.ones(n_b, dtype=np.uint8)
        
        while True:
            if time.time() > deadline - 10.0:
                break
                
            rem_idx = np.where(remaining > 0)[0]
            if len(rem_idx) == 0:
                break
                
            # Calculate effective degree
            deg = (conflict_b * remaining[np.newaxis, :]).sum(axis=1)
            deg[remaining == 0] = n_b + 1
            
            min_deg = deg.min()
            if min_deg >= n_b:
                break
                
            candidates_to_pick = np.where((deg == min_deg) & (remaining > 0))[0]
            if len(candidates_to_pick) == 0:
                break
                
            pick_idx = rng.choice(candidates_to_pick)
            
            selected[pick_idx] = 1
            remaining[pick_idx] = 0
            remaining[conflict_b[pick_idx] == 1] = 0
            
        return selected

    def local_search_b(selected):
        """
        Refine the selected set using 1-for-2 moves.
        """
        sel_indices = np.where(selected == 1)[0]
        if len(sel_indices) == 0:
            return selected, 0
            
        improved = True
        max_moves = 100 # Reduced to allow more restarts
        move_count = 0
        moves_made = 0
        
        while improved and move_count < max_moves and time.time() < deadline - 5.0:
            improved = False
            move_count += 1
            
            sel_indices = np.where(selected == 1)[0]
            k_sel = len(sel_indices)
            
            if k_sel == 0:
                break
            
            # Compute Total_Incompatibility for all nodes against current S
            conflict_with_S = conflict_b[:, sel_indices]
            total_inc = conflict_with_S.sum(axis=1)
            
            # base_mask: nodes compatible with ALL S and not selected
            base_mask = (total_inc == 0) & (selected == 0)
            
            # Try 1-for-2 moves
            order = rng.permutation(k_sel)
            
            for idx in order:
                if time.time() > deadline - 5.0:
                    break
                    
                v_idx_in_S = idx
                v = sel_indices[v_idx_in_S]
                
                # Cond: conflicts ONLY with v
                cond_mask = (total_inc == 1) & (conflict_with_S[:, v_idx_in_S] == 1) & (selected == 0)
                Cand_v = base_mask | cond_mask
                
                cand_indices = np.where(Cand_v > 0)[0]
                
                if len(cand_indices) < 2:
                    continue
                
                sub_conflict = conflict_b[np.ix_(cand_indices, cand_indices)]
                sub_conflict_check = sub_conflict.copy()
                np.fill_diagonal(sub_conflict_check, 1)
                
                if (sub_conflict_check == 0).any():
                    zero_rows, zero_cols = np.where(sub_conflict_check == 0)
                    if len(zero_rows) > 0:
                        u_local = zero_rows[0]
                        w_local = zero_cols[0]
                        u = cand_indices[u_local]
                        w = cand_indices[w_local]
                        
                        selected[v] = 0
                        selected[u] = 1
                        selected[w] = 1
                        improved = True
                        moves_made += 1
                        break
                        
        return selected, moves_made

    # Run multiple restarts
    max_restarts = 100
    for i in range(max_restarts):
        if time.time() > deadline - 20.0:
            break
            
        selected = greedy_construct_b()
        current_count = selected.sum()
        
        selected, moves = local_search_b(selected)
        current_count = selected.sum()
        
        if current_count > best_count_b:
            best_count_b = current_count
            best_selected_b = selected.copy()

    # 4. Combine Results
    # All Type A vectors are valid.
    # Selected Type B vectors are valid.
    # Type A and Type B are mutually compatible.
    
    if best_count_b == 0:
        # Fallback if search failed completely (shouldn't happen)
        return type_a # At least return Type A
    
    selected_type_b = type_b[best_selected_b == 1]
    
    # Concatenate
    result = np.vstack([type_a, selected_type_b])
    
    return result.astype(np.int64)

# EVOLVE-BLOCK-END
