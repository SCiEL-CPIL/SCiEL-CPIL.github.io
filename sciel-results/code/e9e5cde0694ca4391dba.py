"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np


def construct():
    deadline = time.time() + 120  # safety margin before 150s kill
    
    vectors = []
    
    # Unit vectors: norm^2 = 1
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s
            vectors.append(v)
    
    # Weight-2 vectors: norm^2 = 2
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    
    arr = np.array(vectors, dtype=np.int64)
    
    # Verify constraint: min pairwise distance >= max norm
    # Max norm^2 = 2
    # Need all pairwise distances^2 >= 2
    
    return arr
# EVOLVE-BLOCK-END
