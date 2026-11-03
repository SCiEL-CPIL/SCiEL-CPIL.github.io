import numpy as np
import itertools
import time

def construct():
    D = 11
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Generate Type 1 vectors: +/- 2e_i
    # These are mutually compatible and compatible with all Type 2 vectors.
    type1_vectors = []
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            type1_vectors.append(v)
    type1_array = np.array(type1_vectors, dtype=np.int64)
    
    # 2. Generate Type 2 vectors: 4 entries of +/-1, rest 0
    type2_list = []
    for indices in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, idx in enumerate(indices):
                v[idx] = signs[k]
            type2_list.append(v)
            
    type2_array = np.array(type2_list, dtype=np.int32)
    N = len(type2_array)
    
    # 3. Compute Conflict Matrix for Type 2 vectors
    # Conflict if dot product > 2
    # M = V V^T
    # To save memory/time, we can compute in batches or just do it if N is small enough.
    # N = 5280. 5280^2 * 4 bytes = ~110 MB. OK.
    M = type2_array @ type2_array.T
    Conflict = (M > 2).astype(np.uint8)
    np.fill_diagonal(Conflict, 0)
    
    # 4. Greedy MIS on Type 2 graph
    def run_greedy(seed):
        if time.time() > deadline:
            return []
            
        rng = np.random.default_rng(seed)
        active = np.ones(N, dtype=bool)
        # Degree of each node in the active subgraph
        # Initial degree is row sum of Conflict
        deg = Conflict.sum(axis=1).astype(np.int64)
        selected = []
        
        while active.any():
            if time.time() > deadline:
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
            
            # Pick one randomly
            idx = rng.choice(candidates_for_pick)
            
            selected.append(idx)
            
            # Remove idx and its neighbors
            # Neighbors are where Conflict[idx] == 1
            neighbors_mask = Conflict[idx].astype(bool)
            
            # Nodes to remove: idx and neighbors
            # We need to update degrees of remaining active nodes
            # and mark removed nodes as inactive.
            
            # Get list of removed indices
            removed_indices = np.where(neighbors_mask)[0]
            removed_indices = np.concatenate(([idx], removed_indices))
            # Ensure unique and valid
            removed_indices = np.unique(removed_indices)
            
            # Update degrees for active nodes
            # deg[j] -= number of removed neighbors of j
            # This is sum of Conflict[removed_indices, j] for j in active
            if len(removed_indices) > 0:
                # Conflict is symmetric
                removal_counts = Conflict[removed_indices, :].sum(axis=0).astype(np.int64)
                # Update deg only for active nodes
                deg[active] -= removal_counts[active]
                
            # Update active mask
            active[removed_indices] = False
            
        return selected

    best_size = 0
    best_set = []
    
    # Run multiple restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        selected = run_greedy(i)
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected
            
    # 5. Combine results
    if best_size > 0:
        selected_type2 = type2_array[best_set]
        final_vectors = np.vstack([type1_array, selected_type2])
    else:
        # Fallback to just Type 1 if greedy failed
        final_vectors = type1_array
        
    # Optional: Local Search / Improvement
    # Given time constraints, we skip complex local search for now,
    # but the random restarts should suffice.
    
    return final_vectors

if __name__ == "__main__":
    # Test run
    res = construct()
    print(f"Found {len(res)} vectors")
