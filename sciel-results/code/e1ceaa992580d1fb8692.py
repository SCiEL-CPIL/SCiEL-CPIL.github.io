import numpy as np
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    vectors = []
    # Generate all vectors of the form (±1, ±1, 0, ..., 0)
    # These are the roots of the D_11 lattice
    # Count: 11 * 10 * 2 * 2 / 2? No, just iterate.
    for i in range(D):
        for j in range(i + 1, D):
            # 4 sign combinations
            for si in (1, -1):
                for sj in (1, -1):
                    if time.time() > deadline:
                        return np.array(vectors, dtype=np.int64)
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    
    return np.array(vectors, dtype=np.int64)
