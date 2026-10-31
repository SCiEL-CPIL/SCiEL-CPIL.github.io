import itertools
import numpy as np
from math import comb

def construct():
    D = 11
    vectors = []
    # Include all weight-1 vectors (22 vectors)
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s
            vectors.append(v)
    # Include all weight-2 vectors (264 vectors)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                vectors.append(v)
    # Try to add weight-4 vectors where possible
    # For each 4-subset, try to add up to 2 vectors
    # A weight-4 vector has 4 entries of ±1 and rest 0
    # We need to ensure minimum pairwise distance >= max norm
    
    # Let's be conservative and only add weight-4 vectors if they maintain the constraint
    # Actually, let me check: with weight-1 and weight-2 vectors:
    # - Two weight-1 vectors: distance is 0 (if same), 2 (if opposite sign same coord), or sqrt(2) (if different coords)
    #   Wait, if they're different coords with value 1: distance = sqrt(1^2 + 1^2) = sqrt(2)
    #   If same coord, opposite sign: distance = 2
    #   If same coord, same sign: distance = 0 (but we don't have duplicates since each vector is unique)
    #   Actually each vector appears once, so no duplicates.
    #   Two distinct weight-1 vectors: either same coordinate (distance 2) or different (distance sqrt(2))
    # - Two weight-2 vectors: various distances
    # - Weight-1 and weight-2: distance can be small
    
    # Let me compute the max norm: weight-4 vectors have norm 2, weight-2 have norm sqrt(2), weight-1 have norm 1
    # If I include weight-4, max norm becomes 2
    # Then I need all pairwise distances >= 2
    
    # This is a strong constraint. Let me think...
    
    # Actually, the seed program (weight-2 only) has 264 vectors with max norm sqrt(2)
    # Minimum pairwise distance in seed: need to check
    
    # For safety, let me just use weight-1 and weight-2 and verify the constraint
    # Max norm = sqrt(2) for weight-2 vectors
    # Need all pairwise distances >= sqrt(2)
    
    # Let me build the vector list and verify
    arr = np.array(vectors, dtype=np.int64)
    
    # Verify the constraint
    # Compute pairwise distances
    n = len(arr)
    norms = np.linalg.norm(arr, axis=1)
    max_norm = np.max(norms)
    
    # Check minimum pairwise distance
    # This could be slow for large n, but n = 22 + 264 = 286
    # Pairwise distance matrix would be 286x286 = ~81K entries, manageable
    
    # Compute squared distances for efficiency
    sq_norms = np.sum(arr**2, axis=1)
    # Distance squared = |a|^2 + |b|^2 - 2*a.b
    dots = arr @ arr.T
    sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2 * dots
    
    # Set diagonal to inf
    np.fill_diagonal(sq_dists, np.inf)
    min_sq_dist = np.min(sq_dists)
    min_dist = np.sqrt(min_sq_dist)
    
    if min_dist < max_norm - 1e-9:
        # Constraint violated, fall back to weight-2 only
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        arr = np.array(vectors, dtype=np.int64)
    
    return arr
