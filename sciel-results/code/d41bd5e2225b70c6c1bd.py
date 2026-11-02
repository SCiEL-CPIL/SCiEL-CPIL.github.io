"""Structural change: Constructive enumeration of all weight-2 integer vectors in R^11.
Analysis shows that the set of all weight-2 vectors (norm^2=2) forms a valid kissing configuration
of size 220, which is likely optimal or near-optimal for integer vectors.
Adding vectors of other weights requires removing more weight-2 vectors than are gained.
"""
import numpy as np
import itertools


def construct():
    D = 11
    vectors = []
    
    # Generate all weight-2 vectors: two coordinates are +/-1, rest are 0.
    # Number of such vectors: C(11, 2) * 4 = 55 * 4 = 220.
    for i in range(D):
        for j in range(i + 1, D):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * D
                    v[i] = s1
                    v[j] = s2
                    vectors.append(v)
    
    # Convert to numpy array
    result = np.array(vectors, dtype=np.int64)
    
    # Verify validity (optional, for safety)
    # Norm squared is 2 for all.
    # Min distance squared must be >= 2.
    # Dot product <= 1.
    # Since all are distinct, dot product is never 2.
    # Max dot product is 1 (e.g. (1,1,0..) and (1,0,1..) -> dot 1).
    # Min dot product is -2 (e.g. (1,1,0..) and (-1,-1,0..) -> dot -2).
    # Distance squared = 2 + 2 - 2*dot = 4 - 2*dot.
    # If dot = 1, dist^2 = 2. OK.
    # If dot = -2, dist^2 = 8. OK.
    
    return result
