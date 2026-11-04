"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # Generate all integer vectors in R^11 with norm squared <= 4
    # This covers max norms 1, sqrt(2), sqrt(3), 2
    # We will check subsets for each possible max norm
    
    candidates = []
    # Norm squared 1: (1,0,..), (-1,0,..)
    for i in range(D):
        for s in [1, -1]:
            v = [0]*D
            v[i] = s
            candidates.append(v)
            
    # Norm squared 2: (1,1,..), (1,-1,..) etc
    for i, j in itertools.combinations(range(D), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                v = [0]*D
                v[i] = si
                v[j] = sj
                candidates.append(v)
                
    # Norm squared 3: (1,1,1,..) etc
    for i, j, k in itertools.combinations(range(D), 3):
        for si in [1, -1]:
            for sj in [1, -1]:
                for sk in [1, -1]:
                    v = [0]*D
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    candidates.append(v)
                    
    # Norm squared 4:
    # Type 1: (2,0,..)
    for i in range(D):
        for s in [1, -1]:
            v = [0]*D
            v[i] = 2*s
            candidates.append(v)
            
    # Type 2: (1,1,1,1,..)
    for i, j, k, l in itertools.combinations(range(D), 4):
        for si in [1, -1]:
            for sj in [1, -1]:
                for sk in [1, -1]:
                    for sl in [1, -1]:
                        v = [0]*D
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        candidates.append(v)
                        
    # Type 3: (2,2,0,..) -> Norm squared 8, too big? No, 2^2+2^2=8.
    # Wait, norm squared 4 can also be (2, 0, ...) which is covered.
    # Are there others? 1^2+1^2+1^2+1^2=4. 2^2=4.
    # What about (1, 1, 1, 0..)? No, norm sq 3.
    # So candidates list contains all integer vectors with norm squared <= 4.
    
    # Convert to numpy array for fast operations
    C = np.array(candidates, dtype=np.int64)
    n = len(C)
    
    # Precompute squared norms
    norms_sq = np.sum(C * C, axis=1)
    
    best_set = []
    best_score = 0
    
    # Possible max norms squared: 1, 2, 3, 4
    # For a given max_norm_sq, we filter candidates with norm_sq <= max_norm_sq
    # And require pairwise dist_sq >= max_norm_sq
    
    for max_norm_sq in [1, 2, 3, 4]:
        if time.time() > deadline:
            break
            
        # Filter candidates
        mask = norms_sq <= max_norm_sq
        C_sub = C[mask]
        n_sub = len(C_sub)
        
        if n_sub == 0:
            continue
            
        # We want to find the largest subset S of C_sub such that
        # for all u, v in S, u != v, ||u-v||^2 >= max_norm_sq
        
        # Greedy approach: 
        # Randomize order? Or sort by norm?
        # Let's try a simple greedy: pick vectors one by one, check compatibility.
        # To maximize size, we might want to pick "dense" areas? 
        # Actually, for packing, it's often better to just pick as many as possible.
        
        # Since n_sub is small (< 10000), we can afford O(N^2) checks per iteration if we are careful,
        # but N iterations of O(N^2) is O(N^3) which is too slow for N=7000.
        
        # Better: Precompute distance matrix? 7000x7000 is 49M entries. 
        # 49M * 8 bytes = 392 MB. Might be too much memory or slow to compute.
        # But we only need to check against the *current* set.
        
        # Let's use a heuristic: 
        # 1. Shuffle indices
        # 2. Iterate, if compatible with all in current_set, add it.
        
        indices = np.arange(n_sub)
        np.random.shuffle(indices)
        
        current_set = []
        current_set_indices = []
        
        # Optimization: If max_norm_sq is small, many vectors are compatible.
        # If max_norm_sq is large, few are.
        
        for idx in indices:
            v = C_sub[idx]
            compatible = True
            # Check against all in current_set
            if current_set:
                # Vectorized check: dist_sq = ||v - u||^2 = ||v||^2 + ||u||^2 - 2 v.u
                # We need dist_sq >= max_norm_sq
                # ||v - u||^2 = ||v||^2 - 2 v.u + ||u||^2
                # Since ||v||^2 <= max_norm_sq and ||u||^2 <= max_norm_sq,
                # dist_sq >= max_norm_sq  <=>  ||v||^2 + ||u||^2 - 2 v.u >= max_norm_sq
                
                # Get current set vectors
                # For speed, keep current_set as numpy array
                pass # Placeholder for logic below

        # Let's implement the loop properly with numpy arrays for the current set
        current_set_arr = np.empty((0, D), dtype=np.int64)
        
        for idx in indices:
            v = C_sub[idx]
            v_norm_sq = norms_sq[mask][idx]
            
            if len(current_set_arr) == 0:
                current_set_arr = np.vstack([current_set_arr, v])
                continue
            
            # Calculate squared distances from v to all in current_set_arr
            # dist_sq = ||v - u||^2 = ||v||^2 + ||u||^2 - 2 * v @ u
            # u_norm_sq = sum(u^2, axis=1)
            u_norm_sq = np.sum(current_set_arr * current_set_arr, axis=1)
            dots = current_set_arr @ v
            dist_sq = v_norm_sq + u_norm_sq - 2 * dots
            
            if np.min(dist_sq) >= max_norm_sq:
                current_set_arr = np.vstack([current_set_arr, v])
                
        if len(current_set_arr) > best_score:
            best_score = len(current_set_arr)
            best_set = current_set_arr.copy()
            
    # Return the best set found
    if len(best_set) == 0:
        # Fallback to seed if something went wrong
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return best_set
# EVOLVE-BLOCK-END
