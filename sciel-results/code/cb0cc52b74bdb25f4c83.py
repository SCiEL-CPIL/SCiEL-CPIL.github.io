"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

D = 11

# EVOLVE-BLOCK-START
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        if time.time() > deadline:
            break
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vectors.append(v)
        if time.time() > deadline:
            break
            
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
