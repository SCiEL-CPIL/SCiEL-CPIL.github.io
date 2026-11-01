import numpy as np
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    # Generate all vectors of the form e_i + e_j, e_i - e_j, etc.
    # These are the shortest vectors of the D11 lattice (norm squared 2)
    # plus their sign variations.
    # Actually, the set of all vectors with exactly two non-zero entries +-1
    # is the set of shortest vectors of the D11 lattice (if sum is 0)
    # and A11 lattice?
    
    # The seed generates:
    # for i, j in combinations(range(D), 2):
    #   for si in (1, -1):
    #     for sj in (1, -1):
    #       v[i]=si, v[j]=sj
    # This generates 110 * 4 = 440 vectors?
    # No, combinations(range(11), 2) has 55 elements.
    # 55 * 4 = 220 vectors.
    
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    
    # Check time
    if time.time() > deadline:
        return np.array(vectors, dtype=np.int64)
        
    return np.array(vectors, dtype=np.int64)
