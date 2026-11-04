import numpy as np
from itertools import combinations, product

def construct():
    """
    Constructs a set of nonzero integer vectors in R^11 such that the minimum 
    pairwise distance is at least the maximum vector norm.
    
    The strategy is to use all vectors of the form \pm e_i \pm e_j where i < j.
    These vectors have norm sqrt(2).
    For any two distinct vectors u, v in this set, their dot product u.v <= 1.
    Thus, ||u - v||^2 = ||u||^2 + ||v||^2 - 2u.v = 2 + 2 - 2u.v >= 4 - 2(1) = 2.
    So ||u - v|| >= sqrt(2).
    The maximum norm in the set is sqrt(2).
    Therefore, the condition ||u - v|| >= max_norm is satisfied.
    
    The number of such vectors is C(11, 2) * 4 = 55 * 4 = 220.
    """
    D = 11
    vectors = []
    
    # Generate all pairs of indices (i, j) with i < j
    for i, j in combinations(range(D), 2):
        # Generate all 4 sign combinations: (+, +), (+, -), (-, +), (-, -)
        for s1, s2 in product([1, -1], repeat=2):
            v = np.zeros(D, dtype=np.int8)
            v[i] = s1
            v[j] = s2
            vectors.append(v)
    
    return np.array(vectors, dtype=np.int8)
