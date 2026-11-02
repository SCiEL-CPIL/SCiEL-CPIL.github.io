import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin for 150s limit
    
    # Generate ALL nonzero integer vectors with squared norm <= 4
    candidates = []
    
    # Type 1: One 2, rest 0 (squared norm = 4)
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Two 1s, rest 0 (squared norm = 2)
    for idxs in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    # Type 3: Four 1s, rest 0 (squared norm = 4)
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
    
    # Also add vectors with squared norm = 1 (single 1)
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = s
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute dot product matrix
    M = candidates @ candidates.T
    
    # Conflict matrix: 1 if dot > 2 (and i != j)
    # We need min pairwise distance >= max norm
    # For vectors with squared norm <= 4, max norm squared = 4
    # Distance squared = ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j
    # We need this >= 4 for all pairs
    # So: ||v_i||^2 + ||v_j||^2 - 2*M[i,j] >= 4
    # M[i,j] <= (||v_i||^2 + ||v_j||^2 - 4) / 2
    
    # Compute squared norms
    norms_sq = np.sum(candidates ** 2, axis=1)
    
    # Build conflict matrix properly
    # For each pair (i,j), conflict if distance_sq < 4
    # distance_sq = norms_sq[i] + norms_sq[j] - 2*M[i,j]
    # conflict if norms_sq[i] + norms_sq[j] - 2*M[i,j] < 4
    
    norms_sq_matrix = norms_sq[:, None] + norms_sq[None, :]
    dist_sq = norms_sq_matrix - 2 * M
    Conflict = (dist_sq < 4).astype(np.uint8)
    np.fill_diagonal(Conflict, 0)
    
    best_set = []
    best_size = 0
    
    def run_greedy(seed):
        if time.time() > deadline:
            return []
            
        rng = np.random.default_rng(seed)
        active = np.ones(N, dtype=bool)
        deg = Conflict.sum(axis=1).astype(np.int64)
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
            idx = rng.choice(candidates_for_pick)
            
            selected.append(idx)
            
            neighbors_mask = Conflict[idx].astype(bool)
            removed_mask = np.zeros(N, dtype=bool)
            removed_mask[idx] = True
            removed_mask[neighbors_mask & active] = True
            
            removed_indices = np.where(removed_mask)[0]
            
            if len(removed_indices) > 0:
                removal_counts = Conflict[removed_indices, :].sum(axis=0).astype(np.int64)
                deg[active] -= removal_counts[active]
                
            active[removed_mask] = False
            
        return selected

    # Fast local search using vectorized operations
    def local_search_fast(selected_indices):
        if time.time() > deadline:
            return selected_indices
        
        selected_set = set(selected_indices)
        selected_arr = np.array(list(selected_set))
        
        improved = True
        max_iters = 50
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            # Shuffle order
            rng_ls = np.random.default_rng()
            order = rng_ls.permutation(len(selected_arr))
            selected_arr = selected_arr[order]
            
            for i_idx in range(len(selected_arr)):
                if time.time() > deadline:
                    break
                
                i = selected_arr[i_idx]
                
                # Find neighbors of i not in selected_set
                neighbors = np.where(Conflict[i])[0]
                valid_js = [j for j in neighbors if j not in selected_set]
                
                if len(valid_js) < 2:
                    continue
                
                # Vectorized check: which valid_js are compatible with all of selected_set \ {i}
                valid_js_arr = np.array(valid_js)
                
                # For each j in valid_js, check if it conflicts with any k in selected_set \ {i}
                # Create mask for selected_set \ {i}
                rest_mask = np.zeros(N, dtype=bool)
                rest_mask[list(selected_set)] = True
                rest_mask[i] = False
                
                # Conflict counts of each valid_j with the rest
                conflict_with_rest = Conflict[valid_js_arr, :].sum(axis=1)
                compatible_js = valid_js_arr[conflict_with_rest == 0]
                
                if len(compatible_js) < 2:
                    continue
                
                # Find a pair in compatible_js that are mutually compatible
                found = False
                for p in range(len(compatible_js)):
                    if found:
                        break
                    j1 = compatible_js[p]
                    # Check which of compatible_js[p+1:] are compatible with j1
                    rest_compatible = compatible_js[p+1:]
                    if len(rest_compatible) == 0:
                        continue
                    comp_with_j1 = Conflict[j1, rest_compatible] == 0
                    if comp_with_j1.any():
                        j2 = rest_compatible[np.where(comp_with_j1)[0][0]]
                        # Swap i -> j1, j2
                        selected_set.remove(i)
                        selected_set.add(j1)
                        selected_set.add(j2)
                        improved = True
                        found = True
                        break
                
                if found:
                    break
        
        return list(selected_set)

    # Run multiple greedy restarts
    num_restarts = 200
    for s in range(num_restarts):
        if time.time() > deadline:
            break
        
        res = run_greedy(s)
        if not res:
            continue
        
        res_improved = local_search_fast(res)
        
        if len(res_improved) > best_size:
            best_size = len(res_improved)
            best_set = res_improved
            
            # If we've found a good set, do more local search
            if best_size >= 100:
                # Continue with more restarts but also try to improve current best
                pass
    
    if not best_set:
        best_set = list(range(min(22, N)))
        
    return candidates[best_set].astype(np.int64)
