"""Program for kissing number in dimension 11."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np


def construct():
    """
    Construct a set of nonzero integer vectors in R^11 whose minimum pairwise
    distance is at least the maximum vector norm.
    
    Strategy: Use all weight-4 vectors with entries in {-1, 0, 1}.
    - Each vector has norm sqrt(4) = 2
    - Minimum pairwise distance between distinct such vectors is 2
    - Thus min_dist >= max_norm (2 >= 2)
    - Count: C(11,4) * 2^4 = 330 * 16 = 5280
    """
    vectors = []
    
    for coords in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for coord, sign in zip(coords, signs):
                v[coord] = sign
            vectors.append(v)
    
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END

