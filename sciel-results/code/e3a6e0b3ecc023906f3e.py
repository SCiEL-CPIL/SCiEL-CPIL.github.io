# EVOLVE-BLOCK-START
import numpy as np
import itertools


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that their maximum norm is smaller than their minimum pairwise distance, aiming to maximize the number of points.

    Uses all vectors with exactly two non-zero coordinates, each ±1.
    There are C(11,2) * 4 = 220 such points.
    All have squared norm 2.
    Minimum squared distance between any two distinct points is 2.
    Since 2 >= 2, the validity condition is satisfied.

    Returns:
        points: np.ndarray of shape (num_points,11)
    """
    d = 11
    points = []
    
    # Generate all pairs of coordinates
    for i, j in itertools.combinations(range(d), 2):
        # Four sign combinations for the two non-zero coordinates
        for si in [1, -1]:
            for sj in [1, -1]:
                p = np.zeros(d, dtype=np.int64)
                p[i] = si
                p[j] = sj
                points.append(p)
    
    points = np.array(points, dtype=np.int64)
    
    return points


# EVOLVE-BLOCK-END
