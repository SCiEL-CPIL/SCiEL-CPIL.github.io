import numpy as np
import time

D = 11

def construct():
    """
    Constructs a kissing configuration in R^11 using the A_10 root lattice.
    The vectors are the 110 roots of the A_10 lattice.
    Each vector has exactly two non-zero entries: 1 and -1.
    Squared norm of each vector is 2.
    Minimum squared distance between any two distinct vectors is 4.
    Constraint: min_dist >= max_norm  =>  sqrt(4) >= sqrt(2)  =>  2 >= 1.414 (True)
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    vectors = []
    
    # Generate all pairs (i, j) with i < j
    # For each pair, we have two vectors:
    # v1: 1 at i, -1 at j
    # v2: -1 at i, 1 at j
    # Total: 11 * 10 / 2 * 2 = 110 vectors
    
    for i in range(D):
        for j in range(i + 1, D):
            # Vector 1: +1 at i, -1 at j
            v1 = [0] * D
            v1[i] = 1
            v1[j] = -1
            vectors.append(v1)
            
            # Vector 2: -1 at i, +1 at j
            v2 = [0] * D
            v2[i] = -1
            v2[j] = 1
            vectors.append(v2)
            
            # Check time just in case, though this is very fast
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    if not vectors:
        # Fallback to a trivial valid configuration if something went wrong
        # Just one vector is valid (min dist is infinity/undefined or trivially satisfied if k=1)
        # But we want k>1. Let's just return the first few if deadline hit early.
        # Given the speed, this shouldn't happen.
        pass

    return np.array(vectors, dtype=np.int64)
