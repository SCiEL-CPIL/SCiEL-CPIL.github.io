import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Generate pool of vectors with norm squared 3
    # Vectors with exactly three non-zero entries, each +/- 1
    pool = []
    # Indices for the 3 non-zero positions
    indices = list(itertools.combinations(range(D), 3))
    # Signs for the 3 non-zero positions
    signs = list(itertools.product([1, -1], repeat=3))
    
    for idx in indices:
        for s in signs:
            v = [0] * D
            v[idx[0]] = s[0]
            v[idx[1]] = s[1]
            v[idx[2]] = s[2]
            pool.append(v)
            
    pool = np.array(pool, dtype=np.int64)
    n = len(pool)
    
    # Precompute squared distances? 
    # n=1320. n^2 = 1.7M. 
    # Dist matrix is 1.7M * 8 bytes = 13 MB. OK.
    # But we can compute on the fly or use a smarter check.
    
    # Greedy MIS with random restarts
    best_set = []
    best_size = 0
    
    # We will run multiple random restarts
    # Each restart: shuffle pool, then greedily pick valid vectors
    # To speed up, we can maintain a set of selected vectors and check distance
    
    # Optimization: 
    # Instead of checking against all selected, we can just check.
    # Selected size will be ~600.
    # 1320 * 600 checks = 800k distance checks per restart.
    # Each check is vector subtraction and norm. 
    # Vectorized numpy is fast.
    
    for restart in range(200): # Max 200 restarts
        if time.time() > deadline:
            break
            
        # Shuffle
        np.random.shuffle(pool)
        
        selected = []
        selected_arr = None # We will build this incrementally or check against list
        
        # To speed up, we can check against the last few selected? No, must check all.
        # But we can break early if we find a conflict.
        
        for v in pool:
            # Check if v is compatible with all in selected
            is_valid = True
            if selected:
                # Vectorized check against all selected
                # dist_sq = ||v - u||^2 for all u in selected
                # We want min(dist_sq) >= 3
                # v is shape (D,), selected is list of (D,)
                # Convert selected to array? 
                # Doing this inside loop is slow.
                
                # Alternative: keep selected as a list, and check one by one?
                # Or maintain a set of "blocked" vectors?
                # Since we are iterating through pool, we can just check against selected.
                
                # Let's try a simpler check:
                # For each u in selected:
                #   d = np.dot(v-v, v-v)
                #   if d < 3: invalid
                for u in selected:
                    diff = v - u
                    d2 = np.dot(diff, diff)
                    if d2 < 3:
                        is_valid = False
                        break
                        
            if is_valid:
                selected.append(v)
                
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
    # Return the best set found
    if not best_set:
        # Fallback to seed if something goes wrong
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return np.array(best_set, dtype=np.int64)
