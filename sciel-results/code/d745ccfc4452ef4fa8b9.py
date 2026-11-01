"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Generate candidate vectors
    # We consider vectors with entries in {-1, 0, 1}
    # Norm squared 1, 2, or 3.
    # If we include any vector with norm^2 = 3, the max norm^2 is 3.
    # Constraint: min dist^2 >= 3.
    
    candidates = []
    
    # Norm 1 vectors: +-e_i
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
            
    # Norm 2 vectors: +-e_i +-e_j
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
                
    # Norm 3 vectors: +-e_i +-e_j +-e_k
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    candidates.append(v)
                    
    candidates = np.array(candidates)
    num_candidates = len(candidates)
    
    best_set = []
    best_size = 0
    
    # Run greedy multiple times with different random orders
    # We use a simple random permutation
    num_runs = 0
    while time.time() < deadline:
        num_runs += 1
        # Shuffle candidates
        indices = np.random.permutation(num_candidates)
        shuffled_candidates = candidates[indices]
        
        selected = []
        selected_arr = None
        
        for v in shuffled_candidates:
            if selected_arr is None:
                selected.append(v)
                selected_arr = v.reshape(1, -1)
            else:
                # Check distance to all selected vectors
                # dist^2 = ||v - u||^2
                # We need min dist^2 >= 3
                # Note: If we selected a norm 1 vector, and now adding norm 3, 
                # the max norm is 3. So threshold is 3.
                # If we only select norm 2, max norm is 2. Threshold 2.
                # But we are mixing. The threshold is determined by the max norm in the set.
                # Since we include norm 3 vectors, the max norm will likely be 3.
                # So we enforce dist^2 >= 3 for all pairs.
                
                # Optimization: If the set is empty, add.
                # dists = np.sum((selected_arr - v)**2, axis=1)
                # if np.min(dists) >= 3:
                
                # To be slightly faster, we can precompute norms? No, dist is direct.
                diffs = selected_arr - v
                dists_sq = np.sum(diffs**2, axis=1)
                if np.min(dists_sq) >= 3:
                    selected.append(v)
                    selected_arr = np.vstack([selected_arr, v])
                    
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected
            
        # If we reached a very high number, we might stop early?
        # K(11) is ~500. If we get > 480, it's good.
        if best_size > 480:
             # We could break, but let's try to improve a bit more if time permits
             pass
             
    if not best_set:
        # Fallback to the seed solution if greedy failed (unlikely)
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return np.array(best_set, dtype=np.int64)

# EVOLVE-BLOCK-END
