import numpy as np
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates.
    The set consists of all vectors with squared norm 2 (exactly two entries are +/-1,
    the rest are 0).
    
    This configuration achieves k=220.
    Verification:
    - All points have norm squared 2. So max_norm_sq = 2.
    - For any two distinct points u, v in this set:
      - If they have disjoint supports, u.v = 0, ||u-v||^2 = 2+2-0 = 4 >= 2.
      - If they share exactly one index, u.v = +/-1, ||u-v||^2 = 2+2-2(+/-1) = 4 or 6 >= 2.
      - If they share both indices (same support), they differ in signs.
        - If they are opposites (e.g., (1,1) and (-1,-1)), u.v = -2, ||u-v||^2 = 2+2-2(-2) = 8 >= 2.
        - If they differ in one sign (e.g., (1,1) and (1,-1)), u.v = 0, ||u-v||^2 = 2+2-0 = 4 >= 2.
    - Thus min_dist_sq = 2.
    - Condition min_dist_sq >= max_norm_sq holds (2 >= 2).
    """
    d = 11
    points = []
    
    # Iterate over all pairs of indices (i, j) where i < j
    for i, j in combinations(range(d), 2):
        # Iterate over all sign combinations for the two non-zero entries
        for s_i, s_j in product([1, -1], repeat=2):
            vec = np.zeros(d, dtype=np.int64)
            vec[i] = s_i
            vec[j] = s_j
            points.append(vec)
            
    # Convert list of arrays to a single 2D array
    result = np.array(points, dtype=np.int64)
    
    # Ensure the shape is (k, 11)
    # The problem statement says "rounded with np.around and cast to int64".
    # Since we generate integers directly, this is already done, but we 
    # explicitly apply the transformation to strictly follow the contract.
    result = np.around(result).astype(np.int64)
    
    return result
