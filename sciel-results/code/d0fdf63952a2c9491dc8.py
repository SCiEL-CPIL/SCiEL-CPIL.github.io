import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # Generate candidates with squared norm 4
    candidates = []
    
    # Type 1: One 2, rest 0 (squared norm 4)
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four 1s, rest 0 (squared norm 4)
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute dot product matrix
    M = candidates @ candidates.T
    
    # Conflict matrix: 1 if dot > 2 (and i != j)
    # Note: M[i,i] = 4, which is > 2, but we don't conflict with ourselves.
    Conflict = (M > 2).astype(np.uint8)
    np.fill_diagonal(Conflict, 0)
    
    best_set = []
    best_size = 0
    
    # Precompute neighbor lists
    neighbors = [np.where(Conflict[i] == 1)[0] for i in range(N)]
    
    def run_greedy(seed, temperature=0.0):
        """Greedy set packing with random tie-breaking."""
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
            
            # Find all active nodes with degree close to minimum
            if temperature > 0:
                # Soft min: pick from nodes with deg <= min_deg + temperature
                threshold = min_deg + temperature
                candidates_for_pick = active_indices[active_deg <= threshold]
            else:
                candidates_for_pick = active_indices[active_deg == min_deg]
            
            if len(candidates_for_pick) == 0:
                candidates_for_pick = active_indices
                
            idx = rng.choice(candidates_for_pick)
            
            selected.append(idx)
            
            # Remove idx and its neighbors
            neighbors_mask = Conflict[idx].astype(bool)
            removed_mask = np.zeros(N, dtype=bool)
            removed_mask[idx] = True
            removed_mask[neighbors_mask & active] = True
            
            removed_indices = np.where(removed_mask)[0]
            
            # Update degrees
            if len(removed_indices) > 0:
                removal_counts = Conflict[removed_indices, :].sum(axis=0).astype(np.int64)
                deg[active] -= removal_counts[active]
                
            active[removed_mask] = False
            
        return selected
    
    def fast_local_search(selected_indices, max_rounds=50):
        """
        Fast local search using 2-for-1 swaps.
        For each i in S, try to find j1, j2 not in S such that:
        - j1, j2 conflict with i
        - j1, j2 are compatible with each other
        - j1, j2 are compatible with S \ {i}
        If found, replace i with j1, j2.
        """
        if time.time() > deadline:
            return selected_indices
            
        selected_set = set(selected_indices)
        selected_arr = np.array(list(selected_set), dtype=np.int64)
        
        # Precompute for each node in S, the set of nodes that conflict ONLY with that node
        # and are compatible with all other nodes in S
        
        improved = True
        round_num = 0
        while improved and round_num < max_rounds and time.time() < deadline:
            improved = False
            round_num += 1
            
            # Shuffle order for diversity
            indices_list = list(selected_set)
            rng_ls = np.random.default_rng(round_num * 1000 + 42)
            rng_ls.shuffle(indices_list)
            
            for i in indices_list:
                if time.time() > deadline:
                    return list(selected_set)
                
                # Find neighbors of i not in selected_set
                nb_i = neighbors[i]
                valid_js = [j for j in nb_i if j not in selected_set]
                
                if len(valid_js) < 2:
                    continue
                
                valid_js_arr = np.array(valid_js, dtype=np.int64)
                
                # Check which valid_js are compatible with S \ {i}
                # S \ {i} as array
                sel_without_i = np.array([k for k in selected_set if k != i], dtype=np.int64)
                
                if len(sel_without_i) == 0:
                    compatible_with_rest = valid_js_arr
                else:
                    # For each j in valid_js, check if Conflict[j, sel_without_i].sum() == 0
                    # This is a (len(valid_js), len(sel_without_i)) matrix
                    conflicts_matrix = Conflict[valid_js_arr[:, None], sel_without_i[None, :]]
                    compat_mask = (conflicts_matrix.sum(axis=1) == 0)
                    compatible_with_rest = valid_js_arr[compat_mask]
                
                if len(compatible_with_rest) < 2:
                    continue
                
                # Find a compatible pair in compatible_with_rest
                cwr = compatible_with_rest
                found_pair = False
                
                # Try to find j1, j2 in cwr that are compatible with each other
                # Use Conflict matrix to check
                for p in range(len(cwr)):
                    if found_pair:
                        break
                    j1 = cwr[p]
                    # Check all j2 > j1
                    for q in range(p + 1, len(cwr)):
                        j2 = cwr[q]
                        if not Conflict[j1, j2]:
                            # Found a valid swap
                            selected_set.remove(i)
                            selected_set.add(j1)
                            selected_set.add(j2)
                            improved = True
                            found_pair = True
                            break
                
                if found_pair:
                    break  # Restart the pass
            
        return list(selected_set)
    
    # Phase 1: Run greedy with multiple seeds and temperatures
    for s in range(300):
        if time.time() > deadline:
            break
        
        # Vary temperature for diversity
        if s < 100:
            temp = 0.0
        elif s < 200:
            temp = 1.0
        else:
            temp = 2.0
            
        res = run_greedy(s, temperature=temp)
        if not res:
            continue
            
        # Apply local search
        res_improved = fast_local_search(res)
        
        if len(res_improved) > best_size:
            best_size = len(res_improved)
            best_set = res_improved
    
    # Phase 2: Additional local search iterations on best set
    if best_set and time.time() < deadline:
        # Try more aggressive local search
        for _ in range(20):
            if time.time() > deadline:
                break
            improved_set = fast_local_search(best_set, max_rounds=100)
            if len(improved_set) > best_size:
                best_size = len(improved_set)
                best_set = improved_set
    
    if not best_set:
        # Fallback: just take the first 22 candidates (Type 1 vectors)
        best_set = list(range(22))
        
    return candidates[best_set].astype(np.int64)
