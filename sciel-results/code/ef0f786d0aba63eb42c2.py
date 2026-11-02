import numpy as np
import itertools
import time

D = 11

def construct():
    # Generate all integer vectors in R^11 with squared norm 2
    # These are vectors with exactly two non-zero entries, each being +/- 1
    # Number of such vectors: C(11, 2) * 2^2 = 55 * 4 = 220
    
    vectors = []
    
    # Iterate over all pairs of indices
    for i, j in itertools.combinations(range(D), 2):
        # Iterate over all sign combinations for the two entries
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                v = np.zeros(D, dtype=np.int64)
                v[i] = s1
                v[j] = s2
                vectors.append(v)
                
    return np.array(vectors)
