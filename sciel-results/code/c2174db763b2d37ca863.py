# EVOLVE-BLOCK-START
import numpy as np


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that their maximum norm is smaller than their minimum pairwise distance, aiming to maximize the number of points.

    Returns:
        points: np.ndarray of shape (num_points,11)
    """
    d = 11
    points = []
    
    # Generate all integer vectors in Z^11 with squared norm 2.
    # These vectors have exactly two non-zero entries, each being 1 or -1.
    # Total count: C(11, 2) * 4 = 55 * 4 = 220.
    
    for i in range(d):
        for j in range(i + 1, d):
            # Sign combinations for indices i and j
            signs = [
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1)
            ]
            
            for si, sj in signs:
                v = np.zeros(d, dtype=np.int64)
                v[i] = si
                v[j] = sj
                points.append(v)
    
    points_array = np.array(points)
    
    # The task requires rounding and casting to int64, which is already done,
    # but we follow the instruction explicitly.
    points_array = np.around(points_array).astype(np.int64)
    
    return points_array


# EVOLVE-BLOCK-END
