# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def kissing_number11() -> np.ndarray:
    """
    Constructs a kissing configuration in dimension 11 using the minimal vectors
    of the D11 lattice (even unimodular lattice). The D11 lattice has 220 minimal
    vectors, all with squared norm 2, and the minimum squared distance between
    any two distinct minimal vectors is 2, satisfying the condition min_dist^2 >= max_norm^2.

    Returns:
        points: np.ndarray of shape (220, 11) with int64 entries
    """
    d = 11
    vectors = []
    
    # Generate all minimal vectors of D11:
    # For each pair (i, j) with i < j, add 4 vectors with ±1 at positions i and j
    for i, j in combinations(range(d), 2):
        # (1, 1) at (i, j)
        v1 = np.zeros(d, dtype=np.int64)
        v1[i] = 1
        v1[j] = 1
        vectors.append(v1)
        
        # (1, -1) at (i, j)
        v2 = np.zeros(d, dtype=np.int64)
        v2[i] = 1
        v2[j] = -1
        vectors.append(v2)
        
        # (-1, 1) at (i, j)
        v3 = np.zeros(d, dtype=np.int64)
        v3[i] = -1
        v3[j] = 1
        vectors.append(v3)
        
        # (-1, -1) at (i, j)
        v4 = np.zeros(d, dtype=np.int64)
        v4[i] = -1
        v4[j] = -1
        vectors.append(v4)
    
    points = np.array(vectors, dtype=np.int64)
    points = np.around(points).astype(np.int64)
    
    return points


# EVOLVE-BLOCK-END
