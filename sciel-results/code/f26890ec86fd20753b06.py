import numpy as np
import itertools

D = 11

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    Strategy: Use all vectors of weight 2 with entries in {-1, 1}.
    There are C(11, 2) * 4 = 55 * 4 = 220 such vectors.
    For any two distinct vectors u, v of weight 2:
    Max norm squared M^2 = 2.
    Min distance squared:
    - If supports are disjoint, u.v = 0, dist^2 = 2+2-0 = 4 >= 2.
    - If supports share 1 index, u.v in {-1, 1}, dist^2 in {2, 6} >= 2.
    - If supports are identical (same 2 indices), u != v implies u.v in {-2, 0}, 
      dist^2 in {4, 8} >= 2. (u.v=2 only if u=v).
    Thus, all 220 vectors form a valid kissing configuration.
    """
    vectors = []
    # Iterate over all pairs of indices
    for i, j in itertools.combinations(range(D), 2):
        # Iterate over all sign combinations for these two indices
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                v = [0] * D
                v[i] = s1
                v[j] = s2
                vectors.append(v)
    
    return np.array(vectors, dtype=np.int64)
