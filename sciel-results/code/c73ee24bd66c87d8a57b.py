import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    vectors = []
    
    # Type 1: One component is +/- 2, others 0
    for i in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[i] = 2 * sign
            vectors.append(v)
            
    # Type 3: Four components are +/- 1, others 0
    indices = range(D)
    for combo in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, val in zip(combo, signs):
                v[idx] = val
            vectors.append(v)
            
    if time.time() > deadline:
        return np.array(vectors[:100], dtype=np.int64)

    return np.array(vectors, dtype=np.int64)
