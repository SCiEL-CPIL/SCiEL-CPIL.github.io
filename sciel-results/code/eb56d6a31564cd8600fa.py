import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill
    
    # Generate candidates
    candidates = []
    
    # Type 1: One 2, rest 0 (Norm squared = 4)
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four 1s/-1s, rest 0 (Norm squared = 4)
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute dot product matrix
    # N is approx 5300, so N x N is ~28M entries.
    # int32 dot product might overflow? Max dot is 4. 5300*4 is small.
    # But V V^T sums products. Max value in V is 2. Max dot is 4.
    # Using int64 to be safe for intermediate sums if needed, but int32 is fine here.
    M = candidates @ candidates.T
    
    # Conflict matrix: 1 if dot > 2 (and i != j)
    # Norm squared is 4 for all.
    # Distance squared d^2 = 4 + 4 - 2 dot = 8 - 2 dot.
    # We require d^2 >= 4 (max norm squared).
    # 8 - 2 dot >= 4 => 4 >= 2 dot => dot <= 2.
    # So conflict if dot > 2.
    Conflict = (M > 2).astype(np.uint8)
    np.fill_diagonal(Conflict, 0)
    
    # Type 1 vectors (first 2*D = 22 vectors) never conflict with anyone else?
    # Type 1 dot Type 1: 4 if same, 0 if diff.
    # Type 1 dot Type 2: 2 or -2 or 0.
    # So Type 1 vectors only conflict with themselves (diagonal).
    # We can fix all Type 1 vectors in the solution.
    n_type1 = 2 * D
    type1_indices = list(range(n_type1))
    
    # We need to find an independent set in the subgraph of Type 2 vectors.
    # Type 2 indices are n_type1 to N-1.
    # Let's work on the subgraph.
    n_type2 = N - n_type1
    # Map global indices to local indices for Type 2
    # Local 0 corresponds to Global n_type1
    
    # Extract sub-conflict matrix for Type 2
    Conflict_2 = Conflict[n_type1:, n_type1:].copy()
    
    def run_greedy(seed):
        if time.time() > deadline:
            return []
            
        rng = np.random.default_rng(seed)
        active = np.ones(n_type2, dtype=bool)
        # Use int64 for degrees to avoid casting issues
        deg = Conflict_2.sum(axis=1).astype(np.int64)
        selected_local = []
        
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
            
            selected_local.append(idx)
            
            # Remove idx and neighbors
            neighbors_mask = Conflict_2[idx].astype(bool)
            removed_mask = np.zeros(n_type2, dtype=bool)
            removed_mask[idx] = True
            removed_mask[neighbors_mask & active] = True
            
            removed_indices = np.where(removed_mask)[0]
            
            if len(removed_indices) > 0:
                removal_counts = Conflict_2[removed_indices, :].sum(axis=0).astype(np.int64)
                deg[active] -= removal_counts[active]
            
            active[removed_mask] = False
            
        return selected_local

    def local_search(selected_local_indices):
        if time.time() > deadline:
            return list(selected_local_indices)
        
        selected_set = set(selected_local_indices)
        improved = True
        iters = 0
        max_iters = 50
        
        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            iters += 1
            indices_list = list(selected_set)
            rng_ls = np.random.default_rng(iters * 12345 + 7)
            rng_ls.shuffle(indices_list)
            
            for i in indices_list:
                if time.time() > deadline - 5:
                    return list(selected_set)
                
                # Find j not in selected_set that conflicts ONLY with i
                # i.e., j is a neighbor of i, but not a neighbor of any k in selected_set \ {i}
                
                # Get neighbors of i
                neighbors_i = np.where(Conflict_2[i])[0]
                
                # Filter neighbors to those not in selected_set
                valid_js = [j for j in neighbors_i if j not in selected_set]
                
                if len(valid_js) < 2:
                    continue
                
                # For each j in valid_js, check if it conflicts with any k in selected_set \ {i}
                compatible_with_rest = []
                for j in valid_js:
                    # Check conflicts with S \ {i}
                    conflict_count = 0
                    for k in selected_set:
                        if k == i:
                            continue
                        if Conflict_2[j, k]:
                            conflict_count += 1
                            break
                    if conflict_count == 0:
                        compatible_with_rest.append(j)
                
                if len(compatible_with_rest) < 2:
                    continue
                
                # Find a pair (j1, j2) in compatible_with_rest such that Conflict_2[j1, j2] == 0
                found = False
                # To speed up, we can just check the first few or random pairs
                # But compatible_with_rest might be small.
                for a_idx in range(len(compatible_with_rest)):
                    if found:
                        break
                    ja = compatible_with_rest[a_idx]
                    for b_idx in range(a_idx + 1, len(compatible_with_rest)):
                        jb = compatible_with_rest[b_idx]
                        if Conflict_2[ja, jb] == 0:
                            selected_set.remove(i)
                            selected_set.add(ja)
                            selected_set.add(jb)
                            improved = True
                            found = True
                            break
                            
                if improved:
                    break
        
        return list(selected_set)

    best_set_local = []
    best_size = 0
    
    # Multiple random restarts for Greedy
    n_restarts = 50
    for seed in range(n_restarts):
        if time.time() > deadline - 10:
            break
        result = run_greedy(seed)
        if len(result) > best_size:
            best_size = len(result)
            best_set_local = result
            
    # Local search on the best
    if time.time() < deadline - 5:
        improved_set = local_search(best_set_local)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set_local = improved_set
            
    # More restarts if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        for seed in range(n_restarts, n_restarts + 100):
            if time.time() > deadline - 5:
                break
            result = run_greedy(seed)
            if len(result) > best_size:
                best_size = len(result)
                best_set_local = result
                if time.time() < deadline - 5:
                    improved_set = local_search(best_set_local)
                    if len(improved_set) > best_size:
                        best_size = len(improved_set)
                        best_set_local = improved_set

    # Construct final vector list
    # Indices in best_set_local are local to Type 2
    # Global index = n_type1 + local_index
    final_indices = type1_indices + [n_type1 + idx for idx in best_set_local]
    
    vectors = candidates[final_indices]
    
    if len(vectors) == 0:
        # Fallback
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        
    return vectors.astype(np.int64)

if __name__ == "__main__":
    # Quick test
    v = construct()
    print(f"Found {len(v)} vectors")
