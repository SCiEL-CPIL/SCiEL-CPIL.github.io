"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Generate candidate integer vectors
    # We focus on small norms to maximize density
    candidates = []
    
    # Norm sqrt(2): (1, 1, 0, ...)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
                
    # Norm sqrt(5): (2, 1, 0, ...)
    for i, j in itertools.combinations(range(D), 2):
        for s2 in (1, -1):
            for s1 in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = 2 * s2
                v[j] = s1
                candidates.append(v)
                # Also swap positions? No, i,j are distinct indices.
                # The vector (2 at i, 1 at j) is distinct from (1 at i, 2 at j)
                # We need to generate (1 at i, 2 at j) as well
                v2 = np.zeros(D, dtype=np.int64)
                v2[i] = s1
                v2[j] = 2 * s2
                candidates.append(v2)

    # Norm sqrt(8): (2, 2, 0, ...)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = 2 * si
                v[j] = 2 * sj
                candidates.append(v)
                
    # Norm sqrt(9): (3, 0, ...)
    for i in range(D):
        for si in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = 3 * si
            candidates.append(v)

    # Norm sqrt(10): (3, 1, 0, ...)
    for i, j in itertools.combinations(range(D), 2):
        for s3 in (1, -1):
            for s1 in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = 3 * s3
                v[j] = s1
                candidates.append(v)
                v2 = np.zeros(D, dtype=np.int64)
                v2[i] = s1
                v2[j] = 3 * s3
                candidates.append(v2)
                
    # Norm sqrt(13): (3, 2, 0, ...)
    for i, j in itertools.combinations(range(D), 2):
        for s3 in (1, -1):
            for s2 in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = 3 * s3
                v[j] = 2 * s2
                candidates.append(v)
                v2 = np.zeros(D, dtype=np.int64)
                v2[i] = 2 * s2
                v2[j] = 3 * s3
                candidates.append(v2)

    # Remove duplicates (shouldn't be any, but safe)
    # And filter out zero vectors (none generated)
    
    # Convert to list of tuples for hashing
    cand_tuples = [tuple(v) for v in candidates]
    unique_cands = list(set(cand_tuples))
    candidates = [np.array(c, dtype=np.int64) for c in unique_cands]
    
    best_set = []
    best_score = 0
    
    # Precompute norms squared
    norms_sq = [np.dot(v, v) for v in candidates]
    
    # Greedy search with random restarts
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Shuffle candidates
        indices = list(range(len(candidates)))
        np.random.shuffle(indices)
        
        current_set = []
        current_norms_sq = []
        max_norm_sq = 0
        
        for idx in indices:
            if time.time() > deadline:
                break
                
            v = candidates[idx]
            v_norm_sq = norms_sq[idx]
            
            # New max norm if we add v
            new_max_norm_sq = max(max_norm_sq, v_norm_sq)
            
            # Check constraints
            valid = True
            for u_idx in range(len(current_set)):
                u = current_set[u_idx]
                u_norm_sq = current_norms_sq[u_idx]
                
                # Distance constraint: ||v - u|| >= new_max_norm_sq
                # ||v - u||^2 = ||v||^2 + ||u||^2 - 2 v.u
                # We need ||v||^2 + ||u||^2 - 2 v.u >= new_max_norm_sq
                dist_sq = v_norm_sq + u_norm_sq - 2 * np.dot(v, u)
                if dist_sq < new_max_norm_sq:
                    valid = False
                    break
            
            if valid:
                current_set.append(v)
                current_norms_sq.append(v_norm_sq)
                max_norm_sq = new_max_norm_sq
                
        if len(current_set) > best_score:
            best_score = len(current_set)
            best_set = current_set
            # Optional: Try to improve locally?
            # For now, just record.
            
    return np.array(best_set, dtype=np.int64) if best_set else np.array([], dtype=np.int64).reshape(0, D)
# EVOLVE-BLOCK-END
