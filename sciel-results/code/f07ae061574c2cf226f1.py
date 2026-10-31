import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit

    # Generate all weight-4 vectors in {-1, 0, 1}^11
    candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
    
    vectors = np.array(candidates, dtype=np.int64)
    N = len(vectors)  # 5280
    
    if N == 0:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    
    # Compute dot products
    dots = vectors @ vectors.T
    
    # Conflict if dot > 2 (for distinct vectors)
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Greedy MIS with min-degree selection
    def greedy_mis(conflict_matrix, n, deadline, rng_seed=42):
        alive = np.ones(n, dtype=bool)
        selected = []
        degrees = conflict_matrix.sum(axis=1).astype(np.int32)
        rng = np.random.default_rng(rng_seed)
        
        while np.any(alive) and time.time() < deadline:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            selected.append(chosen)
            neighbors = np.where(conflict_matrix[chosen])[0]
            to_remove = np.unique(np.concatenate(([chosen], neighbors)))
            alive[to_remove] = False
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflict_matrix[rem][:, to_remove].sum(axis=1)
        
        return selected
    
    # Fill pass: try to add any unselected vertex compatible with all selected
    def fill_pass(selected_list, conflict_matrix, n, deadline):
        selected_set = set(selected_list)
        selected_arr = np.zeros(n, dtype=bool)
        for s in selected_set:
            selected_arr[s] = True
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if not selected_arr[i]:
                    # Check if i has no conflicts with any selected vertex
                    if not np.any(conflict_matrix[i] & selected_arr):
                        selected_arr[i] = True
                        selected_set.add(i)
                        improved = True
        
        return list(selected_set)
    
    best_selected = []
    restart_deadline = deadline - 5
    
    # Try multiple random restarts with fill pass
    num_restarts = 80
    for restart in range(num_restarts):
        if time.time() > restart_deadline:
            break
        seed = 42 + restart
        selected = greedy_mis(conflicts, N, restart_deadline, rng_seed=seed)
        # Apply fill pass to try to add more vertices
        selected = fill_pass(selected, conflicts, N, restart_deadline)
        if len(selected) > len(best_selected):
            best_selected = selected
    
    result_vectors = vectors[best_selected]
    return result_vectors
