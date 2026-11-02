import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Generate ALL nonzero integer vectors with squared norm = 4 (Weight 4)
    # Entries are +/- 1
    candidates = []
    
    # Type: Four 1s (or -1s), rest 0
    # Number of such vectors: C(11, 4) * 2^4 = 330 * 16 = 5280
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int8)
    N = len(candidates)
    
    # Compute dot product matrix
    # M[i,j] = v_i . v_j
    M = candidates @ candidates.T
    
    # Conflict matrix: 1 if dot > 2
    # We require min pairwise distance >= max norm.
    # Norm^2 = 4 for all.
    # Dist^2 = 4 + 4 - 2*dot = 8 - 2*dot.
    # Dist^2 >= 4 => 8 - 2*dot >= 4 => 2*dot <= 4 => dot <= 2.
    # So conflict if dot > 2.
    Conflict = (M > 2).astype(np.uint8)
    
    # Greedy Independent Set
    def run_greedy(seed):
        if time.time() > deadline:
            return []
            
        rng = np.random.default_rng(seed)
        active = np.ones(N, dtype=bool)
        # Degree of each node in the conflict graph
        deg = Conflict.sum(axis=1).astype(np.int32)
        selected = []
        
        while active.any():
            if time.time() > deadline:
                break
            
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
                
            active_deg = deg[active_indices]
            min_deg = np.min(active_deg)
            candidates_for_pick = active_indices[active_deg == min_deg]
            
            # Random pick among min-degree nodes to diversify
            idx = rng.choice(candidates_for_pick)
            
            selected.append(idx)
            
            # Update active set and degrees
            # Neighbors of idx are those with Conflict[idx, j] == 1
            neighbors = Conflict[idx].astype(bool)
            
            # Nodes to remove: idx and its neighbors
            remove_mask = np.zeros(N, dtype=bool)
            remove_mask[neighbors] = True
            remove_mask[idx] = True
            
            # Update degrees for active nodes
            # For each active node j, if it was connected to idx, its degree decreases by 1
            # deg[j] -= Conflict[idx, j]
            
            # Vectorized update on active subset
            # We need to update deg for all active nodes
            # conflict_with_idx = Conflict[idx, :]
            # deg[active] -= conflict_with_idx[active]
            
            # Note: deg is global. We update only active ones.
            # But Conflict[idx, active] is a vector of 0s and 1s.
            
            active_deg_update = Conflict[idx, active_indices].astype(np.int32)
            # We need to map back to global indices or just update the global array directly?
            # Updating global array directly is safer.
            # deg[active_indices] = deg[active_indices] - active_deg_update
            
            deg[active_indices] -= active_deg_update
            
            active[remove_mask] = False
            
        return selected

    def local_search_add(selected_indices):
        if time.time() > deadline:
            return selected_indices
            
        is_selected = np.zeros(N, dtype=bool)
        for i in selected_indices:
            is_selected[i] = True
            
        selected_list = list(selected_indices)
        
        improved = True
        # Limit iterations to avoid infinite loops if logic is flawed, though it should converge
        max_iters = 200 
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            # Compute number of conflicts each node has with the currently selected set
            # counts[j] = sum(Conflict[j, selected_list])
            # If counts[j] == 0 and not is_selected[j], we can add j.
            
            if len(selected_list) == 0:
                # If empty, we can add any node. But greedy should not return empty if N>0.
                # Just add first node.
                is_selected[0] = True
                selected_list.append(0)
                improved = True
                continue

            # Vectorized sum
            # Conflict is N x N. selected_list is indices.
            # Conflict[:, selected_list] is N x k
            # Sum along axis 1 gives N x 1
            counts = Conflict[:, selected_list].sum(axis=1)
            
            # Find addable nodes
            addable_mask = (~is_selected) & (counts == 0)
            addable_indices = np.where(addable_mask)[0]
            
            if len(addable_indices) > 0:
                # Heuristic: Pick the one with the smallest degree in the addable set?
                # Or just the first one.
                # To be safe and fast, pick the first one.
                # Actually, picking one that allows MORE future additions is better.
                # But that requires lookahead.
                # For now, random or first.
                idx = addable_indices[0]
                
                is_selected[idx] = True
                selected_list.append(idx)
                improved = True
            else:
                break
                
        return selected_list

    best_set = []
    best_size = 0
    
    # Run multiple greedy restarts
    num_restarts = 500  # Increase restarts since each is fast
    
    for s in range(num_restarts):
        if time.time() > deadline:
            break
        
        res = run_greedy(s)
        if not res:
            continue
        
        # Local search to fill holes
        res_improved = local_search_add(res)
        
        if len(res_improved) > best_size:
            best_size = len(res_improved)
            best_set = res_improved
            
    if not best_set:
        # Fallback: just return a few valid vectors
        # Basis vectors scaled by 2?
        # v = [2,0,...] norm 2.
        # Dist between [2,0] and [0,2] is sqrt(8) > 2.
        # Dist between [2,0] and [-2,0] is 4 > 2.
        # So we can take all 22 vectors of type (2 at one pos, +/-).
        best_set = list(range(22))
        
    return candidates[best_set].astype(np.int64)
