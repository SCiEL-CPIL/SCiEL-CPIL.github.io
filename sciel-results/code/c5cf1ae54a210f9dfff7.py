import numpy as np
import itertools
import time

def construct():
    D = 11
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # 1. Generate candidate vectors with squared norm 4
    # Type 1: +/- 2e_i (11 * 2 = 22 vectors)
    # Type 2: 4 entries of +/-1, rest 0 (C(11,4) * 2^4 = 330 * 16 = 5280 vectors)
    
    type1_vectors = []
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            type1_vectors.append(v)
    type1_array = np.array(type1_vectors, dtype=np.int64)
    
    type2_list = []
    for indices in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, idx in enumerate(indices):
                v[idx] = signs[k]
            type2_list.append(v)
            
    type2_array = np.array(type2_list, dtype=np.int32)
    N = len(type2_array)
    
    # 2. Compute Conflict Matrix for Type 2 vectors
    # Two vectors conflict if their distance is < max_norm = 2.
    # dist^2 = ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b = 4 + 4 - 2a.b = 8 - 2a.b
    # We need dist >= 2, so dist^2 >= 4.
    # 8 - 2a.b >= 4  =>  2a.b <= 4  =>  a.b <= 2.
    # So conflict if a.b > 2.
    
    M = type2_array @ type2_array.T
    Conflict = (M > 2).astype(np.uint8)
    np.fill_diagonal(Conflict, 0)
    
    # 3. Greedy Maximum Independent Set (MIS) on Type 2 graph
    # We want to maximize the number of non-conflicting Type 2 vectors.
    # Then add all Type 1 vectors, which are compatible with each other and with all Type 2 vectors.
    
    def run_greedy(seed):
        if time.time() > deadline:
            return []
            
        rng = np.random.default_rng(seed)
        active = np.ones(N, dtype=bool)
        
        # Degree of each node in the active subgraph
        # Initially, degree is the number of conflicts for each node
        deg = Conflict.sum(axis=1).astype(np.int64)
        
        selected = []
        
        while active.any():
            if time.time() > deadline - 5:
                break
            
            # Find active nodes
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
                
            # Get degrees of active nodes
            active_deg = deg[active_indices]
            
            # Find min degree
            min_deg = np.min(active_deg)
            
            # Find all active nodes with min degree
            candidates_for_pick = active_indices[active_deg == min_deg]
            
            # Pick one randomly to break symmetry
            idx = rng.choice(candidates_for_pick)
            
            selected.append(idx)
            
            # Remove idx and its neighbors from active set
            neighbors_mask = Conflict[idx].astype(bool)
            neighbors_mask[idx] = True
            
            # Get list of removed indices
            removed_indices = np.where(neighbors_mask)[0]
            
            # Update degrees for remaining active nodes
            # deg[j] -= number of removed neighbors of j
            if len(removed_indices) > 0:
                # Conflict is symmetric, so Conflict[removed_indices, j] gives count of removed neighbors for j
                removal_counts = Conflict[removed_indices, :].sum(axis=0).astype(np.int64)
                # Update deg only for active nodes
                # We must be careful with types and indices
                active_mask = active.copy()
                active_mask[removed_indices] = False
                # Actually, we just subtract from all, but only active ones matter for future.
                # However, we need to update `deg` for the next iteration's `active_deg` calculation.
                # `deg` is global. We can just subtract from all.
                deg -= removal_counts
                
            # Update active mask
            active[removed_indices] = False
            
        return selected

    best_size = 0
    best_set = []
    
    # Run multiple restarts
    num_restarts = 200
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        selected = run_greedy(i)
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected
            
    # 4. Combine results
    if best_size > 0:
        selected_type2 = type2_array[best_set]
        final_vectors = np.vstack([type1_array, selected_type2])
    else:
        final_vectors = type1_array
        
    return final_vectors

if __name__ == "__main__":
    # Test run
    res = construct()
    print(f"Found {len(res)} vectors")
