import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit (evaluator kills at 150s)
    
    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    # Also try weight-2 vectors with entries in {-1, 1} (Norm^2 = 2)
    # These can coexist with weight-4 vectors if dot product is small enough
    v2_candidates = []
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v2_candidates.append(v)
    
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    vectors_v2 = np.array(v2_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    N1 = len(vectors_v1)
    N2 = len(vectors_v2)
    
    # Compute conflicts within v4 set
    dots_v4 = vectors_v4 @ vectors_v4.T
    conflicts_v4 = dots_v4 > 2  # distance^2 = 8 - 2*dot, need dist^2 >= max_norm^2
    # For norm^2=4 vectors: dist^2 = 4+4-2*dot = 8-2*dot >= 4 means dot <= 2
    # So conflict when dot > 2
    np.fill_diagonal(conflicts_v4, False)
    
    # Compute conflicts between v1 and v4
    # v1 has norm^2=4, v4 has norm^2=4
    # dist^2 = 8 - 2*dot >= 4 means dot <= 2
    dots_v1_v4 = vectors_v1 @ vectors_v4.T
    conflicts_v1_v4 = dots_v1_v4 > 2
    
    # Compute conflicts between v1 and v1
    dots_v1 = vectors_v1 @ vectors_v1.T
    conflicts_v1 = dots_v1 > 2
    np.fill_diagonal(conflicts_v1, False)
    
    # For v2 vectors (norm^2=2), they're smaller. 
    # If we include them, max_norm^2 might be 4 (from v4/v1)
    # dist^2 between v2 and v4 = 2+4-2*dot = 6-2*dot >= 4 means dot <= 1
    # dist^2 between v2 and v1 = 2+4-2*dot = 6-2*dot >= 4 means dot <= 1
    # dist^2 between v2 and v2 = 4-2*dot >= 4 means dot <= 0
    dots_v2_v4 = vectors_v2 @ vectors_v4.T
    conflicts_v2_v4 = dots_v2_v4 > 1
    
    dots_v2_v1 = vectors_v2 @ vectors_v1.T
    conflicts_v2_v1 = dots_v2_v1 > 1
    
    dots_v2 = vectors_v2 @ vectors_v2.T
    conflicts_v2 = dots_v2 > 0
    np.fill_diagonal(conflicts_v2, False)
    
    best_total_count = 0
    best_v4_mask = np.zeros(N4, dtype=bool)
    best_v1_mask = np.zeros(N1, dtype=bool)
    best_v2_mask = np.zeros(N2, dtype=bool)
    
    rng = np.random.default_rng(42)
    
    # Strategy: First find good v4+v1 set, then try to add v2 vectors
    # Run multiple greedy MIS attempts on the combined v4+v1 graph
    
    # Build combined adjacency for v4 and v1
    # Total nodes = N4 + N1
    total_n = N4 + N1
    combined_conflicts = np.zeros((total_n, total_n), dtype=bool)
    combined_conflicts[:N4, :N4] = conflicts_v4
    combined_conflicts[N4:, N4:] = conflicts_v1
    combined_conflicts[:N4, N4:] = conflicts_v1_v4.T
    combined_conflicts[N4:, :N4] = conflicts_v1_v4
    
    degrees = combined_conflicts.sum(axis=1).astype(np.int32).copy()
    
    max_restarts = 20
    for r in range(max_restarts):
        if time.time() > deadline - 15:
            break
            
        seed = 42 + r * 137
        rng_r = np.random.default_rng(seed)
        
        alive = np.ones(total_n, dtype=bool)
        selected_indices = []
        deg = degrees.copy()
        
        while np.any(alive) and time.time() < deadline - 10:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            # Random tie-breaking for min degree
            min_deg = np.min(deg[alive_indices])
            cands = alive_indices[deg[alive_indices] == min_deg]
            if len(cands) == 0:
                break
            chosen = cands[rng_r.integers(len(cands))]
            
            selected_indices.append(chosen)
            
            neighbors = np.where(combined_conflicts[chosen])[0]
            to_remove = np.unique(np.concatenate(([chosen], neighbors)))
            
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                deg[rem] -= combined_conflicts[rem][:, to_remove].sum(axis=1)
                
        # Fast Local Search: Try to add random unselected nodes
        selected_mask = np.zeros(total_n, dtype=bool)
        selected_mask[selected_indices] = True
        
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            unselected = np.where(~selected_mask)[0]
            if len(unselected) == 0:
                break
            # Pick a random candidate from unselected
            u = unselected[rng_r.integers(len(unselected))]
            # Check if it conflicts with any currently selected node
            if not np.any(selected_mask & combined_conflicts[u]):
                selected_mask[u] = True
        
        count_v4_v1 = np.sum(selected_mask[:N4]) + np.sum(selected_mask[N4:])
        
        # Now try to add v2 vectors
        v4_sel = selected_mask[:N4]
        v1_sel = selected_mask[N4:]
        
        # For each v2 vector, check if it's compatible with all selected v4 and v1
        v2_compatible = np.ones(N2, dtype=bool)
        for j in range(N2):
            if time.time() > deadline - 5:
                break
            # Check against selected v4
            if np.any(conflicts_v2_v4[j] & v4_sel):
                v2_compatible[j] = False
                continue
            # Check against selected v1
            if np.any(conflicts_v2_v1[j] & v1_sel):
                v2_compatible[j] = False
                continue
        
        # Now greedily select v2 vectors that don't conflict with each other
        v2_selected = []
        v2_alive = v2_compatible.copy()
        v2_deg = np.zeros(N2, dtype=np.int32)
        for j in range(N2):
            if v2_compatible[j]:
                v2_deg[j] = np.sum(conflicts_v2[j] & v2_compatible)
        
        while np.any(v2_alive) and time.time() < deadline - 3:
            alive_v2 = np.where(v2_alive)[0]
            if len(alive_v2) == 0:
                break
            min_deg = np.min(v2_deg[alive_v2])
            cands = alive_v2[v2_deg[alive_v2] == min_deg]
            if len(cands) == 0:
                break
            chosen = cands[rng_r.integers(len(cands))]
            v2_selected.append(chosen)
            
            neighbors = np.where(conflicts_v2[chosen])[0]
            to_remove = np.unique(np.concatenate(([chosen], neighbors)))
            v2_alive[to_remove] = False
            
            rem = np.where(v2_alive)[0]
            if len(rem) > 0:
                v2_deg[rem] -= conflicts_v2[rem][:, to_remove].sum(axis=1)
        
        total_count = count_v4_v1 + len(v2_selected)
        
        if total_count > best_total_count:
            best_total_count = total_count
            best_v4_mask = v4_sel.copy()
            best_v1_mask = v1_sel.copy()
            best_v2_mask = np.zeros(N2, dtype=bool)
            best_v2_mask[v2_selected] = True

    # Build final result
    result_v4 = vectors_v4[best_v4_mask]
    result_v1 = vectors_v1[best_v1_mask]
    result_v2 = vectors_v2[best_v2_mask]
    
    parts = [result_v4, result_v1, result_v2]
    final_vectors = np.vstack([p for p in parts if len(p) > 0])
    
    return final_vectors
