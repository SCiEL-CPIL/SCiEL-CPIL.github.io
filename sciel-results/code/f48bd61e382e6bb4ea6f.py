import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    D = 11
    
    # Generate candidates with squared norms 1, 2, 4
    candidates = []
    
    # Type 1: Norm squared 1 (one 1)
    for i in range(D):
        for s in [1, -1]:
            v = np.zeros(D, dtype=np.int32)
            v[i] = s
            candidates.append(v)
            
    # Type 2: Norm squared 2 (two 1s)
    for idxs in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = np.zeros(D, dtype=np.int32)
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
                
    # Type 3: Norm squared 4 (four 1s OR one 2)
    # One 2
    for i in range(D):
        for s in [1, -1]:
            v = np.zeros(D, dtype=np.int32)
            v[i] = 2 * s
            candidates.append(v)
            
    # Four 1s
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int32)
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute dot product matrix
    M = candidates @ candidates.T
    
    # Compute squared norms
    norms_sq = np.sum(candidates ** 2, axis=1)
    
    # We target R^2 = 4 (max norm squared in our candidate pool)
    # Condition: ||u - v||^2 >= 4
    # ||u||^2 + ||v||^2 - 2 u.v >= 4
    # u.v <= (||u||^2 + ||v||^2 - 4) / 2
    
    # Build conflict matrix
    # Conflict if u.v > threshold
    norms_sq_matrix = norms_sq[:, None] + norms_sq[None, :]
    thresholds = (norms_sq_matrix - 4) / 2.0
    
    # Conflict if M[i,j] > thresholds[i,j]
    # Note: thresholds can be negative.
    # If threshold is -0.5, and M is integer, M > -0.5 means M >= 0.
    # If threshold is 0.5, M > 0.5 means M >= 1.
    # If threshold is 1.0, M > 1.0 means M >= 2.
    
    Conflict = (M > thresholds).astype(np.uint8)
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

    def local_search_add(selected_list):
        if time.time() > deadline:
            return selected_list
            
        selected_set = set(selected_list)
        improved = True
        max_iters = 20
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            iters += 1
            
            # Compute conflicts with selected set
            if selected_set:
                sel_arr = np.array(list(selected_set))
                conflicts_with_selected = Conflict[sel_arr, :].max(axis=0).astype(bool)
            else:
                conflicts_with_selected = np.zeros(N, dtype=bool)
                
            # Candidates: not in selected, not conflicting
            candidates_mask = ~conflicts_with_selected
            candidates_mask[list(selected_set)] = False
            candidate_indices = np.where(candidates_mask)[0]
            
            if len(candidate_indices) == 0:
                break
                
            # Try adding one
            rng = np.random.default_rng()
            rng.shuffle(candidate_indices)
            
            for cidx in candidate_indices:
                selected_list.append(cidx)
                selected_set.add(cidx)
                improved = True
                break
                
        return selected_list

    # Run multiple greedy restarts
    num_restarts = 100
    for s in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        res = run_greedy(s * 100 + 42)
        if not res:
            continue
        
        res_improved = local_search_add(res)
        
        if len(res_improved) > best_size:
            best_size = len(res_improved)
            best_set = res_improved
            
    if not best_set:
        best_set = list(range(min(22, N)))
        
    return candidates[best_set].astype(np.int64)
