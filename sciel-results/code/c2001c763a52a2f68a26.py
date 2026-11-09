# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional integer points with squared norm 3,
    such that the minimum squared pairwise distance is at least the maximum squared norm.
    
    All integer vectors with exactly three non-zero coordinates (each ±1) have squared norm 3.
    For any two distinct such vectors, the minimum squared distance is 4 (when they share
    the same support and differ in exactly one sign). Since 4 >= 3, the condition is satisfied.
    
    Returns:
        points: np.ndarray of shape (num_points, 11) with int64 entries
    """
    d = 11
    # Generate all combinations of 3 indices out of 11
    indices = list(range(d))
    points = []
    
    for combo in combinations(indices, 3):
        i, j, k = combo
        # For each combination of indices, generate all 8 sign combinations
        for si in [-1, 1]:
            for sj in [-1, 1]:
                for sk in [-1, 1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    p[k] = sk
                    points.append(p)
    
    points_array = np.array(points, dtype=np.int64)
    return points_array


# EVOLVE-BLOCK-END
