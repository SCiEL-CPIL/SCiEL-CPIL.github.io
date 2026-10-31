import numpy as np
import itertools

D = 11

def construct():
    vectors = []
    # Generate all vectors with exactly 4 non-zero entries, each +/- 1
    # Choose 4 positions out of 11
    for positions in itertools.combinations(range(D), 4):
        # Generate all 2^4 = 16 sign combinations
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            vectors.append(v)
    
    return np.array(vectors, dtype=np.int64)
