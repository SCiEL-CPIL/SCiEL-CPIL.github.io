"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np


def construct():
    vectors = []
    
    # Type 1: One entry is +/-2, rest 0
    for i in range(D):
        for s in (2, -2):
            v = [0] * D
            v[i] = s
            vectors.append(v)
            
    # Type 2: Four entries are +/-1, rest 0
    # Choose 4 indices
    for indices in itertools.combinations(range(D), 4):
        # Choose signs for these 4 indices
        # There are 2^4 = 16 sign combinations
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(indices, signs):
                v[idx] = s
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
