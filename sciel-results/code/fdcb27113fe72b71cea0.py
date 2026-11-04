import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = 150.0
    
    vectors = []
    
    # Generate all weight 3 vectors
    # Support: combinations of 3 indices from 11
    # Signs: 2^3 = 8 combinations
    for support in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)
            
    # Convert to array
    if not vectors:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    return np.array(vectors, dtype=np.int64)
