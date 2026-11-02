import numpy as np
import itertools
import time

D = 11

def construct():
    # The optimal configuration for integer vectors in R^11 with the given constraint
    # is the set of roots of the D_11 lattice, which are vectors with exactly two
    # non-zero entries, each being +/-1.
    # There are binom(11, 2) * 4 = 55 * 4 = 220 such vectors.
    # For any two distinct such vectors, the dot product is in {-2, -1, 0, 1}.
    # Max norm squared is 2.
    # Min distance squared = 2 + 2 - 2*max_dot = 4 - 2(1) = 2.
    # Constraint: min_dist >= max_norm => 2 >= 2. Satisfied.
    
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            # Signs: (1, 1), (1, -1), (-1, 1), (-1, -1)
            vectors.append([0]*D)
            vectors[-1][i] = 1
            vectors[-1][j] = 1
            
            vectors.append([0]*D)
            vectors[-1][i] = 1
            vectors[-1][j] = -1
            
            vectors.append([0]*D)
            vectors[-1][i] = -1
            vectors[-1][j] = 1
            
            vectors.append([0]*D)
            vectors[-1][i] = -1
            vectors[-1][j] = -1
            
    return np.array(vectors, dtype=np.int64)
