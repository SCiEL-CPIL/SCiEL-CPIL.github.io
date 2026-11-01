import numpy as np
import itertools
import time

def construct():
    D = 11
    candidates = []
    
    # Type 1: two ±1's, rest 0 (norm^2 = 2)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
                
    # Type 2: one ±2, rest 0 (norm^2 = 4)
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            candidates.append(v)
            
    # Type 3: four ±1's, rest 0 (norm^2 = 4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    # Also prepare the pure norm-2 set as a fallback
    norms_sq = np.sum(candidates**2, axis=1)
    mask_t1 = norms_sq == 2
    candidates_t1 = candidates[mask_t1]
    n_t1 = len(candidates_t1)
    
    # The condition for a set S is:
    # min_{u,v in S, u!=v} ||u-v||^2 >= max_{w in S} ||w||^2
    
    # Case 1: S contains only norm^2=2 vectors.
    # max_norm_sq = 2.
    # min_dist_sq >= 2.
    # As analyzed, all distinct norm^2=2 vectors have min_dist_sq >= 2.
    # So S = candidates_t1 is valid. Size = 220.
    
    # Case 2: S contains at least one norm^2=4 vector.
    # max_norm_sq = 4.
    # min_dist_sq >= 4.
    # We need to find a large independent set in the graph where edges connect vectors with dist_sq < 4.
    
    dots = candidates @ candidates.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflict = (dist_sq < 4)
    np.fill_diagonal(conflict, False)
    
    best_selected = []
    best_size = 0
    
    # Fallback
    if n_t1 > best_size:
        best_size = n_t1
        best_selected = np.where(mask_t1)[0].tolist()
        
    # Greedy + Local Search for Case 2
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    np.random.seed(42)
    
    n_attempts = 0
    while time.time() < deadline:
        n_attempts += 1
        # Greedy
        remaining = np.ones(n, dtype=bool)
        degrees = conflict.sum(axis=1).copy()
        selected = np.zeros(n, dtype=bool)
        
        while remaining.any():
            rem_indices = np.where(remaining)[0]
            if len(rem_indices) == 0:
                break
            rem_degrees = degrees[rem_indices]
            min_deg = np.min(rem_degrees)
            min_mask = rem_degrees == min_deg
            min_indices_local = np.where(min_mask)[0]
            
            chosen_local = np.random.choice(min_indices_local)
            chosen_global = rem_indices[chosen_local]
            
            selected[chosen_global] = True
            neigh_mask = conflict[chosen_global]
            to_remove_mask = np.zeros(n, dtype=bool)
            to_remove_mask[chosen_global] = True
            to_remove_mask[neigh_mask] = True
            actual_remove = to_remove_mask & remaining
            
            if np.any(actual_remove):
                deg_decrease = conflict[actual_remove][:, remaining].sum(axis=0)
                degrees[remaining] -= deg_decrease
            remaining[actual_remove] = False
            
        current_size = np.sum(selected)
        
        # Local Search: Try to swap one selected with one unselected to increase size
        # Or add one unselected if it doesn't conflict? (Greedy should have added it if no conflict)
        # So we look for swaps: remove u, add v, if v doesn't conflict with (selected \ {u})
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Get list of selected and unselected
            sel_indices = np.where(selected)[0]
            un_sel_indices = np.where(~selected)[0]
            
            if len(un_sel_indices) == 0:
                break
                
            # For each selected u, find neighbors in un_selected that could replace it?
            # Actually, standard MIS local search:
            # Try to add any vertex from un_selected that has no neighbors in selected.
            # (Greedy missed it? Unlikely if greedy is perfect, but greedy is heuristic)
            
            # Check if any un_selected vertex is independent from selected
            # conflict[un_sel_indices][:, sel_indices].any(axis=1)
            # This is O(N_un * N_sel * D).
            
            # Simpler: Just try random swaps.
            # Pick random u from selected, random v from un_selected.
            # If v does not conflict with any selected except u, and u conflicts with v?
            # Condition: v is not in conflict[selected \ {u}]
            
            for _ in range(100): # Limit iterations per phase
                if time.time() > deadline:
                    break
                u = np.random.choice(sel_indices)
                v = np.random.choice(un_sel_indices)
                
                # Check if v conflicts with any selected node other than u
                # neighbors_of_v_in_selected = np.where(conflict[v] & selected)[0]
                # if u in neighbors_of_v_in_selected and len(neighbors_of_v_in_selected) == 1:
                #    # v only conflicts with u among selected.
                #    # If we remove u, v can be added. Size stays same.
                #    # We want to INCREASE size.
                #    # So this swap is neutral.
                #    pass
                
                # To increase size, we need to remove k nodes and add m > k nodes.
                # This is hard.
                
                # Alternative: Just run more greedy attempts.
                break
        
        if current_size > best_size:
            best_size = current_size
            best_selected = np.where(selected)[0].tolist()
            
    result = candidates[best_selected]
    return result

