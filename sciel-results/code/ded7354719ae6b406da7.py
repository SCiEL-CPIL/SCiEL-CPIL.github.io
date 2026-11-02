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
    # Use symmetry breaking: first non-zero coordinate is positive
    type2 = []
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            # Symmetry breaking: first sign must be positive
            if signs[0] != 1:
                continue
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
    
    # Strategy: Greedy min-degree selection on the reduced symmetry-broken set
    def greedy_min_degree(conflict_mat, N):
        """Greedy selection by minimum degree in remaining graph."""
        selected = np.zeros(N, dtype=bool)
        count = 0
        remaining = np.ones(N, dtype=bool)
        degrees = np.sum(conflict_mat, axis=1)
        
        while np.any(remaining):
            # Find remaining vertex with minimum degree
            min_deg = np.inf
            best_idx = -1
            for i in range(N):
                if remaining[i] and degrees[i] < min_deg:
                    min_deg = degrees[i]
                    best_idx = i
                    if min_deg == 0:
                        break
            
            if best_idx == -1:
                break
            
            # Select this vertex
            selected[best_idx] = True
            count += 1
            remaining[best_idx] = False
            
            # Update degrees of remaining vertices
            # Vertices that conflict with best_idx get their degree decreased
            # (since best_idx is no longer in the remaining graph)
            # Actually, we need to remove best_idx from the conflict graph
            # So for each remaining vertex, if it conflicts with best_idx, its degree decreases by 1
            for j in range(N):
                if remaining[j] and conflict_mat[j, best_idx]:
                    degrees[j] -= 1
        
        return selected, count
    
    # Phase 1: Single greedy min-degree pass
    selected, count = greedy_min_degree(conflict, N2)
    if count > best_count:
        best_count = count
        best_mask = selected.copy()
    
    # Phase 2: Multiple greedy restarts with random tie-breaking
    n_restarts = 200
    for r in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        # Random perturbation of the greedy order
        order = rng.permutation(N2)
        selected = np.zeros(N2, dtype=bool)
        count = 0
        
        for idx in order:
            if not np.any(conflict[idx] & selected):
                selected[idx] = True
                count += 1
        
        if count > best_count:
            best_count = count
            best_mask = selected.copy()
    
    # Phase 3: Local search - try to add compatible unselected vectors
    if time.time() < deadline - 5:
        unselected = np.where(~best_mask)[0]
        for idx in unselected:
            if time.time() > deadline - 3:
                break
            if not np.any(conflict[idx] & best_mask):
                best_mask[idx] = True
                best_count += 1
    
    # Construct final array
    # Type 1 vectors are always compatible with Type 2
    # And mutually compatible.
    type1_arr = np.array(type1, dtype=np.int64)
    type2_selected = type2[best_mask].astype(np.int64)
    
    final_vectors = np.vstack([type1_arr, type2_selected])
    
    return final_vectors
