"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    # Deadline guard
    start_time = time.time()
    timeout = 110  # 10s margin
    
    # 1. Generate Norm sqrt(2) vectors (Squared Norm 2)
    # These are always valid as a set (min dist sqrt(2) >= max norm sqrt(2))
    vecs_sq2 = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vecs_sq2.append(v)
    arr_sq2 = np.array(vecs_sq2, dtype=np.int64)
    best_set = arr_sq2
    best_size = len(arr_sq2)
    
    # 2. Generate Norm sqrt(3) vectors (Squared Norm 3)
    # Check if we can find a larger independent set
    vecs_sq3 = []
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i], v[j], v[k] = si, sj, sk
                    vecs_sq3.append(v)
    arr_sq3 = np.array(vecs_sq3, dtype=np.int64)
    
    if time.time() - start_time > timeout:
        return best_set
        
    # 3. Solve MIS for Norm sqrt(3)
    # Conflict if dot product >= 2
    # Dot product matrix
    # arr_sq3 is (N, 11)
    # Dot = arr_sq3 @ arr_sq3.T
    N = len(arr_sq3)
    
    # To save time, we can run multiple random restarts
    # Precompute conflict structure?
    # Dot product matrix is N x N. N=1320.
    # 1320x1320 int64 is ~14MB.
    
    try:
        dot_prod = arr_sq3 @ arr_sq3.T
        # Conflict if dot_prod > 2 (since we want dot <= 1)
        # Note: diagonal is 3, so self-conflict.
        conflict_mask = dot_prod > 2
        
        # Greedy MIS with random restarts
        best_mis_size = 0
        best_mis_indices = []
        
        num_restarts = 20
        for r in range(num_restarts):
            if time.time() - start_time > timeout:
                break
                
            active_mask = np.ones(N, dtype=bool)
            selected_indices = []
            
            while np.any(active_mask):
                # Pick a random active node
                # To be faster, we can pick the one with fewest active neighbors (Min Degree)
                # But random is simpler. Let's try Min Degree for better results.
                
                # Count active neighbors for each active node
                # active_neighbors_count = conflict_mask[:, active_mask].sum(axis=1)
                # This is expensive per step.
                
                # Simple Random
                active_indices = np.where(active_mask)[0]
                idx = active_indices[np.random.randint(len(active_indices))]
                
                selected_indices.append(idx)
                
                # Remove idx and its neighbors
                active_mask[conflict_mask[idx]] = False
                active_mask[idx] = False
                
            if len(selected_indices) > best_mis_size:
                best_mis_size = len(selected_indices)
                best_mis_indices = selected_indices
                
    except Exception:
        pass
        
    if best_mis_size > best_size:
        best_set = arr_sq3[best_mis_indices]
        
    return best_set
# EVOLVE-BLOCK-END
