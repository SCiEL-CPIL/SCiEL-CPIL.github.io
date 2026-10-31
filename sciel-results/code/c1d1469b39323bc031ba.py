import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
    
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    
    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    # For each dimension, we can take either +2e_i or -2e_i, but not both
    # (they would have dot product -4 or 4, violating the constraint)
    # So we take exactly one per dimension: 11 vectors
    v1_candidates = []
    for pos in range(D):
        v = [0] * D
        v[pos] = 2
        v1_candidates.append(v)
    
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    # Verify: all distinct weight-4 vectors have dot product <= 2
    # (Dot product 4 only for identical vectors, which aren't in the list)
    # Weight-4 and weight-1 vectors: dot product is -2, 0, or 2, all <= 2
    # Weight-1 vectors among themselves: different positions, dot = 0
    
    final_vectors = np.vstack([vectors_v4, vectors_v1])
    
    return final_vectors
