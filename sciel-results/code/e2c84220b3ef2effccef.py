import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    
    vectors = []
    
    # 1. Vectors with exactly 4 non-zero entries (+/- 1)
    indices = range(D)
    for combo in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, val in zip(combo, signs):
                v[idx] = val
            vectors.append(v)
            
    # 2. Vectors with exactly 1 non-zero entry (+/- 2)
    for idx in indices:
        v1 = [0] * D
        v1[idx] = 2
        vectors.append(v1)
        
        v2 = [0] * D
        v2[idx] = -2
        vectors.append(v2)
        
    vectors = np.array(vectors, dtype=np.int64)
    
    return vectors
