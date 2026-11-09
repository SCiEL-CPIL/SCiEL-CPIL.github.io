import numpy as np
import itertools
import time

def generate_norm2_vectors(d=11):
    """
    Generate all integer vectors in Z^d with squared norm 2.
    These are vectors with exactly two non-zero entries, each +/- 1.
    """
    vecs = []
    # Choose 2 indices out of d
    for idxs in itertools.combinations(range(d), 2):
        i, j = idxs
        # Choose signs for i and j
        for si in [1, -1]:
            for sj in [1, -1]:
                v = np.zeros(d, dtype=np.int64)
                v[i] = si
                v[j] = sj
                vecs.append(v)
    return np.array(vecs)

def validate_configuration(points):
    """
    Validate the configuration according to the task contract.
    Returns True if valid, False otherwise.
    """
    if points.shape[1] != 11:
        return False
    k = points.shape[0]
    if k < 2:
        return False
    
    # Check no zero vector
    if np.any(np.all(points == 0, axis=1)):
        return False
        
    # Calculate max squared norm
    norms_sq = np.sum(points**2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Calculate min squared distance
    min_dist_sq = float('inf')
    for i in range(k):
        for j in range(i + 1, k):
            diff = points[i] - points[j]
            d_sq = np.sum(diff**2)
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
                
    # Condition: min_dist_sq >= max_norm_sq
    return min_dist_sq >= max_norm_sq

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580
    
    # Strategy: Use the set of minimal vectors of the D11 lattice.
    # These are all vectors with squared norm 2 (two +/- 1s, rest 0).
    # There are 2 * C(11, 2) = 220 such vectors.
    # For this set:
    #   Max Norm Squared = 2
    #   Min Dist Squared = 2 (between vectors sharing an index with opposite signs)
    #   Condition 2 >= 2 holds.
    #   K = 220.
    
    points = generate_norm2_vectors(11)
    
    # Ensure dtype is int64 and rounded
    points = np.around(points).astype(np.int64)
    
    # Verify validity
    if validate_configuration(points):
        return points
    else:
        # Fallback: Return a known valid small configuration (e.g., +/- e_1)
        # This should never happen for the norm-2 set, but it's safe.
        fallback = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.int64)
        return fallback

# Optional: Try to improve further if time permits?
# Given the structural nature, the norm-2 set is a hard bound for "minimal vectors" in D11.
# Searching for larger sets in other norms is risky and computationally heavy.
# The prompt suggests a structural change. Exact construction of D11 minimal vectors is the most robust improvement over greedy search on N=3/4.

# Note: Can we add vectors of norm 1?
# If we add e1 (norm 1), max norm becomes 2.
# Dist(e1, e1+e2) = 1. 1 < 2. Fails.
# So we cannot mix norm 1 and norm 2.
# Can we mix norm 2 and norm 4?
# If we add 2*e1 (norm 4), max norm 4.
# Dist(2*e1, e1+e2) = (1, -1) -> 2. 2 < 4. Fails.
# So mixing norms generally breaks the condition because the larger norm vector gets "too close" to the smaller ones.
# Thus, the optimal sets are likely homogeneous in norm.
# Norm 1: k=22.
# Norm 2: k=220.
# Norm 3: Need to solve a coding theory problem. Unlikely to exceed 220 quickly.
# Norm 4: Even sparser.

# Therefore, k=220 is likely the global optimum or very close to it for integer lattices in this setting.
