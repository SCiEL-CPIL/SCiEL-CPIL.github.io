# EVOLVE-BLOCK-START
import numpy as np


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that their maximum norm is smaller than their minimum pairwise distance, aiming to maximize the number of points.
    
    This function returns the 110 shortest non-zero vectors of the A_11 lattice.
    These vectors are of the form e_i - e_j and e_j - e_i for 1 <= i < j <= 11.
    They all have squared norm 2.
    The minimum squared distance between any two distinct vectors is 2.
    Thus, min_dist_sq (2) >= max_norm_sq (2).
    
    Returns:
        points: np.ndarray of shape (110, 11)
    """
    d = 11
    points = []
    
    # Generate all vectors e_i - e_j and e_j - e_i
    for i in range(d):
        for j in range(i + 1, d):
            # Vector v1: 1 at i, -1 at j
            v1 = np.zeros(d, dtype=np.int64)
            v1[i] = 1
            v1[j] = -1
            points.append(v1)
            
            # Vector v2: -1 at i, 1 at j
            v2 = np.zeros(d, dtype=np.int64)
            v2[i] = -1
            v2[j] = 1
            points.append(v2)
            
    points_array = np.array(points, dtype=np.int64)
    
    return points_array

# EVOLVE-BLOCK-END
