import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110 # 10s safety margin
    
    # 1. Generate Weight 4 Vectors
    # Indices: combinations of 4 from 11
    # Signs: 4 bits -> 16 combinations
    idx_combinations = list(itertools.combinations(range(D), 4))
    sign_combinations = list(itertools.product([1, -1], repeat=4))
    
    wt4_vectors = []
    for idxs in idx_combinations:
        for signs in sign_combinations:
            v = np.zeros(D, dtype=np.int64)
            for i, s in zip(idxs, signs):
                v[i] = s
            wt4_vectors.append(v)
            
    wt4_matrix = np.array(wt4_vectors, dtype=np.int64)
    N4 = len(wt4_matrix)
    
    # 2. Greedy Selection for Weight 4
    # Try multiple random orders to find a large independent set
    best_selected_indices = []
    best_size = 0
    
    # We can also just do one pass, but random restarts are cheap
    for attempt in range(20):
        if time.time() > deadline: break
        
        order = np.random.permutation(N4)
        available = np.ones(N4, dtype=bool)
        selected = []
        
        for idx in order:
            if not available[idx]:
                continue
            
            # Check conflict with already selected
            # We can maintain a list of selected vectors
            # But checking against all selected every time is O(K*N)
            # Better: update available mask
            # When we select idx, we compute conflicts with all available
            # and mark them as unavailable.
            
            selected.append(idx)
            # Compute dot products of wt4_matrix[idx] with all available
            # dot = wt4_matrix[available] @ wt4_matrix[idx]
            # This is expensive if done inside the loop for every idx.
            # Total cost sum_{k=1}^{K} (N - k) ~ N*K/2.
            # If K ~ 2000, N=5280, ops ~ 5e6 per attempt.
            # 20 attempts ~ 1e8 ops. NumPy is fast, but let's optimize.
            
            # Actually, we can just compute dot products for the current idx against all *currently available* vectors.
            # But we need to update the available mask.
            
            # Let's just do the vectorized check against the whole matrix of *previously selected*?
            # No, the standard greedy MIS on a graph:
            # 1. Pick a vertex.
            # 2. Remove it and its neighbors.
            # Here, "neighbors" are those with dot > 2.
            
            # Implementation:
            # current_v = wt4_matrix[idx]
            # dots = wt4_matrix[available] @ current_v
            # conflict_mask = dots > 2
            # available_indices_list = np.where(available)[0]
            # available[available_indices_list[conflict_mask]] = False
            # available[idx] = False
            
            # This is O(N) per selection. Total O(N*K).
            # With K~2000, N=5280, 10^7 operations per attempt.
            # 20 attempts -> 2*10^8 operations.
            # NumPy dot product is very fast. This should be fine.
            
            pass # Placeholder for logic

    # Let's refine the greedy loop
    best_indices = None
    for attempt in range(50): # More attempts, still fast
        if time.time() > deadline: break
        order = np.random.permutation(N4)
        available = np.ones(N4, dtype=bool)
        selected = []
        
        for idx in order:
            if not available[idx]:
                continue
            selected.append(idx)
            available[idx] = False
            
            # Find conflicts among remaining available
            # Vectorized dot product
            # Only compute for those that are still available
            avail_indices = np.where(available)[0]
            if len(avail_indices) == 0: break
            
            dots = wt4_matrix[avail_indices] @ wt4_matrix[idx]
            conflicts = dots > 2
            
            # Mark conflicts as unavailable
            # avail_indices[conflicts] are the indices to remove
            available[avail_indices[conflicts]] = False
            
        if len(selected) > len(best_indices) if best_indices is not None else len(selected) > 0:
            best_indices = selected
            if len(selected) > 4000: # Heuristic: if we get very high, stop
                 break

    # 3. Add Weight 2 Vectors
    idx_combinations_2 = list(itertools.combinations(range(D), 2))
    sign_combinations_2 = list(itertools.product([1, -1], repeat=2))
    wt2_vectors = []
    for idxs in idx_combinations_2:
        for signs in sign_combinations_2:
            v = np.zeros(D, dtype=np.int64)
            for i, s in zip(idxs, signs):
                v[i] = s
            wt2_vectors.append(v)
    wt2_matrix = np.array(wt2_vectors, dtype=np.int64)
    N2 = len(wt2_matrix)
    
    selected_wt2 = []
    if best_indices:
        selected_wt4 = wt4_matrix[best_indices]
        # For each wt2 vector, check against selected_wt4
        # Condition: u @ v <= 1
        # And against other selected wt2: u @ u' <= 0
        
        for i in range(N2):
            if time.time() > deadline: break
            u = wt2_matrix[i]
            
            # Check against wt4
            # dots_wt4 = selected_wt4 @ u
            # if np.any(dots_wt4 > 1): continue
            
            # Check against selected wt2
            # if selected_wt2:
            #    dots_wt2 = np.array(selected_wt2) @ u
            #    if np.any(dots_wt2 > 0): continue
            
            # To make this faster, we can just do it. N2=220.
            # selected_wt4 size ~2000. 220*2000 = 440k ops. Trivial.
            
            valid = True
            dots_wt4 = selected_wt4 @ u
            if np.any(dots_wt4 > 1):
                valid = False
            
            if valid and selected_wt2:
                dots_wt2 = np.array(selected_wt2) @ u
                if np.any(dots_wt2 > 0):
                    valid = False
            
            if valid:
                selected_wt2.append(u)
                
    # Combine
    if best_indices:
        final_wt4 = wt4_matrix[best_indices]
    else:
        final_wt4 = np.empty((0, D), dtype=np.int64)
        
    if selected_wt2:
        final_wt2 = np.array(selected_wt2, dtype=np.int64)
    else:
        final_wt2 = np.empty((0, D), dtype=np.int64)
        
    result = np.vstack([final_wt4, final_wt2])
    return result
