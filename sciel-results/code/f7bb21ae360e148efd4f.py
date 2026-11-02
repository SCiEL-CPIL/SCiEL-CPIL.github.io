import numpy as np
import time
import itertools

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Generate candidate vectors
    # We focus on sparse vectors with small norms
    candidates = []
    
    # Norm squared 2: e_i +/- e_j
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                candidates.append((2, v))
                
    # Norm squared 3: e_i +/- e_j +/- e_k
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i], v[j], v[k] = si, sj, sk
                    candidates.append((3, v))
                    
    # Norm squared 4: e_i +/- e_j +/- e_k +/- e_l
    for i, j, k, l in itertools.combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = [0] * D
                        v[i], v[j], v[k], v[l] = si, sj, sk, sl
                        candidates.append((4, v))
                        
    # Norm squared 4: 2e_i
    for i in range(D):
        for si in (1, -1):
            v = [0] * D
            v[i] = 2 * si
            candidates.append((4, v))

    # Norm squared 5: e_i +/- e_j +/- e_k +/- e_l +/- e_m
    for i, j, k, l, m in itertools.combinations(range(D), 5):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        for sm in (1, -1):
                            v = [0] * D
                            v[i], v[j], v[k], v[l], v[m] = si, sj, sk, sl, sm
                            candidates.append((5, v))
                            
    # Norm squared 5: 2e_i +/- e_j
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = sj
                candidates.append((5, v))
                # Also 2e_j +/- e_i
                v2 = [0] * D
                v2[j] = 2 * sj
                v2[i] = si
                candidates.append((5, v2))

    # Sort candidates by norm, then by tuple representation for determinism
    candidates.sort(key=lambda x: (x[0], x[1]))
    
    # Initialize with a strong base set
    # Start with norm 2 vectors
    S = []
    R_max = 0
    
    for norm_sq, v in candidates:
        if time.time() > deadline:
            break
            
        # Check if we can add v
        # New max norm
        new_R_max = max(R_max, norm_sq)
        
        if not S:
            S.append(v)
            R_max = new_R_max
            continue
            
        # Convert S to numpy array for fast distance calculation
        # Optimization: Only convert if S changes? 
        # For simplicity, convert each time. S is small (< 1000).
        S_arr = np.array(S, dtype=np.int64)
        v_arr = np.array(v, dtype=np.int64)
        
        # Calculate squared distances
        # dist^2 = sum((S - v)^2)
        diffs = S_arr - v_arr
        dists_sq = np.sum(diffs * diffs, axis=1)
        min_dist_sq = np.min(dists_sq)
        
        if min_dist_sq >= new_R_max:
            S.append(v)
            R_max = new_R_max
            
    return np.array(S, dtype=np.int64)
# EVOLVE-BLOCK-END
