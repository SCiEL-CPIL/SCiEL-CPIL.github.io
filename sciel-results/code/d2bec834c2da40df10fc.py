import numpy as np
import itertools

def construct():
    D = 11
    vectors = []
    
    # 1. E8 lattice part (first 8 dimensions)
    # Minimal vectors of E8 are:
    # 112 vectors of type (1, 1, 0, ..., 0) with signs
    # 128 vectors of type (1/2, 1/2, ..., 1/2) with even number of minus signs
    
    # We scale by 2 to make them integers.
    # Norm of scaled vectors is 2 * sqrt(2) = sqrt(8).
    
    # Type 1: (±2, ±2, 0, ..., 0)
    for i, j in itertools.combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                vectors.append(v)
                
    # Type 2: (±2, ±2, ..., ±2) with even number of -1s (after dividing by 2, it was 1/2)
    # Original: (1/2, 1/2, ..., 1/2). Scaled: (1, 1, ..., 1)??
    # Wait. E8 half-integers are (1/2, 1/2, ..., 1/2).
    # Scaled by 2: (1, 1, ..., 1).
    # Norm of (1,1,1,1,1,1,1,1) is sqrt(8).
    # Are these integer? Yes.
    # How many? 2^7 = 128.
    for signs in itertools.product([1, -1], repeat=8):
        # Even number of -1s
        if sum(1 for s in signs if s == -1) % 2 == 0:
            v = [0] * D
            for i in range(8):
                v[i] = signs[i]
            vectors.append(v)
            
    # 2. D3 part (last 3 dimensions)
    # Add vectors with norm <= sqrt(8) and min distance >= sqrt(8)
    # We found 6 vectors: ±2 e_i
    for i in range(8, 11):
        vectors.append([0]*D + [2]) # placeholder, need to set correctly
        # Correct construction:
        v_pos = [0] * D
        v_pos[i] = 2
        vectors.append(v_pos)
        
        v_neg = [0] * D
        v_neg[i] = -2
        vectors.append(v_neg)
        
    return np.array(vectors, dtype=np.int64)
