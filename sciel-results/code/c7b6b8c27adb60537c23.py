import numpy as np
import itertools
import time

D = 11

def construct():
    deadline = time.time() + 110  # 10s margin
    
    # Strategy 1: Norm sqrt(2) vectors (e_i +/- e_j)
    # These are all mutually compatible.
    vecs_sq2 = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                vecs_sq2.append(v)
    
    vecs_sq2 = np.array(vecs_sq2, dtype=np.int64)
    k1 = len(vecs_sq2)
    
    # Strategy 2: Norm 2 vectors (2*e_i)
    # Compatible with each other, but incompatible with full vecs_sq2.
    # If we use only these, k=22.
    
    # Strategy 3: Norm sqrt(3) vectors (e_i+e_j+e_k)
    # Try to find a large independent set.
    # Total 1320.
    # Greedy approach.
    vecs_sq3_candidates = []
    for i, j, k in itertools.combinations(range(D), 3):
        for signs in itertools.product((1, -1), repeat=3):
            v = np.zeros(D, dtype=np.int64)
            v[i] = signs[0]
            v[j] = signs[1]
            v[k] = signs[2]
            vecs_sq3_candidates.append(v)
            
    vecs_sq3_candidates = np.array(vecs_sq3_candidates, dtype=np.int64)
    
    # Greedy independent set on vecs_sq3_candidates
    # Condition: u . v <= 1
    selected_sq3 = []
    used = set()
    
    # Shuffle for better greedy?
    # For speed, just iterate.
    # To be faster, we can skip if time is tight.
    if time.time() > deadline - 5:
        return vecs_sq2
    
    # Simple greedy
    for v in vecs_sq3_candidates:
        is_valid = True
        for u in selected_sq3:
            if np.dot(u, v) > 1:
                is_valid = False
                break
        if is_valid:
            selected_sq3.append(v)
            
    if time.time() > deadline - 2:
        return vecs_sq2
        
    vecs_sq3 = np.array(selected_sq3, dtype=np.int64) if selected_sq3 else np.array([], dtype=np.int64)
    k2 = len(vecs_sq3)
    
    # Compare
    if k2 > k1:
        return vecs_sq3
    else:
        return vecs_sq2

# EVOLVE-BLOCK-END
