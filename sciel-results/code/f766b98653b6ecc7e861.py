import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Initialize with the seed set (D_11 roots)
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vectors.append(tuple(v))
    
    vec_list = list(vectors)
    
    # Candidate pool: All integer vectors with norm squared <= 5
    # This includes e_i (norm 1), e_i+e_j (norm 2), e_i+2e_j (norm 5), etc.
    candidates = []
    # Generate small integer vectors
    for coords in itertools.product(range(-2, 3), repeat=D):
        if any(c != 0 for c in coords):
            n2 = sum(c*c for c in coords)
            if n2 <= 5:
                candidates.append(tuple(coords))
    
    # Remove existing vectors from candidates
    existing_set = set(vec_list)
    candidates = [c for c in candidates if c not in existing_set]
    
    # Greedy addition
    # Sort candidates by norm squared (smaller norm is easier to fit?)
    # Or just iterate.
    
    added_count = 0
    for c in candidates:
        if time.time() > deadline:
            break
            
        n2_c = sum(x*x for x in c)
        
        # Check if c can be added
        # Condition: for all v in vec_list, ||v - c||^2 >= max_norm_sq
        # Current max norm sq is 2 (from seed).
        # If we add c with n2_c > 2, max_norm_sq becomes n2_c.
        # This makes the constraint harder for existing pairs too?
        # No, existing pairs are already valid for max_norm_sq=2.
        # If max_norm_sq increases, we must re-check ALL pairs?
        # That is expensive.
        
        # Strategy: Only add vectors with norm squared <= 2 to keep max_norm_sq = 2.
        # Candidates with n2 <= 2 are e_i and e_i +/- e_j.
        # e_i +/- e_j are already in the set.
        # So only e_i are new candidates with n2 <= 2.
        
        if n2_c > 2:
            continue
            
        # Check compatibility with all existing vectors
        compatible = True
        for v in vec_list:
            diff_sq = 0
            for k in range(D):
                diff = v[k] - c[k]
                diff_sq += diff * diff
            # Need diff_sq >= 2 (max_norm_sq)
            if diff_sq < 2:
                compatible = False
                break
        
        if compatible:
            vec_list.append(c)
            added_count += 1
            
    return np.array(vec_list, dtype=np.int64)
