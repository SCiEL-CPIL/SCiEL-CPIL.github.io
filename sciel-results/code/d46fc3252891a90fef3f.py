# EVOLVE-BLOCK-START
import numpy as np
import itertools

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that their minimum pairwise squared distance is >= their maximum squared norm.
    
    We use all points with squared norm 2.
    Max norm squared = 2.
    Min pairwise distance squared = 2.
    Condition 2 >= 2 holds.
    Number of points = 4 * C(11, 2) = 220.
    """
    d = 11
    points = []
    for i in range(d):
        for j in range(i + 1, d):
            # Signs: (+,+), (+,-), (-,+), (-,-)
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = s1
                    p[j] = s2
                    points.append(p)
    
    points_array = np.array(points, dtype=np.int64)
    return points_array

# EVOLVE-BLOCK-END
