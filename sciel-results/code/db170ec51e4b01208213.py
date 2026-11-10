# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def kissing_number11() -> np.ndarray:
    """
    Constructs 11-dimensional integer points (coordinates in {-1,0,1} with
    Hamming weight exactly 2) such that the minimum pairwise squared distance
    (which is 2) is >= the maximum squared norm (which is 2).

    There are C(11,2) = 55 position pairs, each with 3 sign patterns
    ((+1,+1), (+1,-1), (-1,-1)), giving 165 points.

    Returns:
        points: np.ndarray of shape (165, 11) of int64
    """
    d = 11
    vecs = []
    for (i, j) in combinations(range(d), 2):
        # three sign patterns for the two non-zero coordinates
        for a, b in ((1, 1), (1, -1), (-1, -1)):
            v = [0] * d
            v[i] = a
            v[j] = b
            vecs.append(v)

    points = np.array(vecs, dtype=np.int64)
    return points


# EVOLVE-BLOCK-END
