import itertools
import numpy as np
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    vectors = []
    # Base set: D_11 minimal vectors
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vectors.append(v)
    
    vecs = np.array(vectors, dtype=np.int64)
    
    # Generate pool of weight-3 vectors
    pool = []
    for idxs in itertools.combinations(range(D), 3):
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    v = [0] * D
                    v[idxs[0]] = s1
                    v[idxs[1]] = s2
                    v[idxs[2]] = s3
                    pool.append(v)
    
    pool = np.array(pool, dtype=np.int64)
    
    # Greedy addition
    # We check if a pool vector can be added to the current set
    # Condition: dot product <= 1 for all existing vectors
    # Note: Norms are sqrt(2) for base, sqrt(3) for pool.
    # Max norm becomes sqrt(3).
    # Base-Base dist >= 2 >= sqrt(3). OK.
    # Base-Pool dist >= sqrt(3) iff dot <= 1.
    # Pool-Pool dist >= sqrt(3) iff dot <= 1.
    
    current_vecs = vecs
    current_norm_sq_max = 3  # sqrt(3)^2
    
    # To speed up, we can precompute dot products?
    # Pool size 1320, Current size 220.
    # 1320 * 220 = 290k dot products. Fast.
    
    added = []
    for v in pool:
        # Check against current_vecs
        # v is shape (D,)
        # current_vecs is shape (N, D)
        dots = current_vecs @ v
        if np.all(dots <= 1):
            # Check against added
            if added:
                added_vecs = np.array(added)
                dots_added = added_vecs @ v
                if np.all(dots_added <= 1):
                    added.append(v)
        
        if time.time() > deadline:
            break
            
    if added:
        new_vecs = np.vstack([current_vecs, np.array(added)])
        return new_vecs
        
    return current_vecs
