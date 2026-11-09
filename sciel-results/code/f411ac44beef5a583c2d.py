# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional integer lattice points with the property that
    the minimum squared pairwise distance is >= the maximum squared norm.
    
    Starts with all weight-2 vectors (220 points), then greedily adds more points
    from a candidate pool while maintaining the distance constraint.
    """
    deadline = time.time() + 600.0
    safety_margin = 30.0  # seconds
    
    d = 11
    
    # Generate base set: all weight-2 vectors
    base_points = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in (-1, 1):
                for sj in (-1, 1):
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    base_points.append(p)
    
    points = np.array(base_points, dtype=np.int64)
    k = len(points)
    
    # Current max norm squared
    max_norm_sq = 2  # all weight-2 vectors have norm^2 = 2
    # Current min pairwise distance squared
    min_dist_sq = 4  # verified for weight-2 vectors
    
    # Generate candidate pool: weight-3 vectors
    candidates = []
    for combo in combinations(range(d), 3):
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                for s3 in (-1, 1):
                    p = np.zeros(d, dtype=np.int64)
                    p[combo[0]] = s1
                    p[combo[1]] = s2
                    p[combo[2]] = s3
                    candidates.append(p)
    
    candidates = np.array(candidates, dtype=np.int64)
    
    # For each candidate, check if it can be added
    # A candidate c can be added if:
    # 1. ||c||^2 <= min_dist_sq (new max norm^2 = max(old_max, ||c||^2) must be <= min_dist_sq)
    #    Actually, new max_norm_sq = max(2, ||c||^2) = ||c||^2 for weight-3 (norm^2=3)
    #    So we need all existing points to be at distance >= 3 from c,
    #    AND all existing pairs to still have distance >= 3 (they do, min is 4)
    # 2. For all existing points p: ||p - c||^2 >= max(2, ||c||^2) = 3
    #    Since ||c||^2 = 3, we need ||p - c||^2 >= 3 for all p in current set.
    
    added_count = 0
    for idx in range(len(candidates)):
        if time.time() > deadline - safety_margin:
            break
        
        c = candidates[idx]
        c_norm_sq = int(np.dot(c, c))
        if c_norm_sq == 0:
            continue
        
        new_max_norm = max(max_norm_sq, c_norm_sq)
        
        # Check distance from c to all existing points
        diffs = points - c  # shape (k, d)
        dists_sq = np.sum(diffs * diffs, axis=1)
        
        if np.min(dists_sq) < new_max_norm:
            continue
        
        # Also need to check that new_max_norm <= current min_dist_sq
        # because adding c might reduce min_dist_sq
        # New min_dist_sq = min(old_min_dist_sq, min_dist_from_c)
        # We need new_min_dist_sq >= new_max_norm
        new_min_dist = min(min_dist_sq, int(np.min(dists_sq)))
        if new_min_dist < new_max_norm:
            continue
        
        # Can add this candidate
        points = np.vstack([points, c])
        k += 1
        max_norm_sq = new_max_norm
        min_dist_sq = new_min_dist
        added_count += 1
    
    # Verify final result
    k = len(points)
    assert k >= 2
    assert points.shape[1] == 11
    
    # No zero vector
    for p in points:
        assert np.any(p != 0), "Zero vector found"
    
    # Check min pairwise distance^2 >= max norm^2
    max_norm_sq_final = 0
    for p in points:
        ns = int(np.dot(p, p))
        if ns > max_norm_sq_final:
            max_norm_sq_final = ns
    
    min_dist_sq_final = float('inf')
    for i in range(k):
        for j in range(i + 1, k):
            diff = points[i] - points[j]
            ds = int(np.dot(diff, diff))
            if ds < min_dist_sq_final:
                min_dist_sq_final = ds
    
    assert min_dist_sq_final >= max_norm_sq_final, f"min_dist_sq={min_dist_sq_final} < max_norm_sq={max_norm_sq_final}"
    
    # Round and cast as required
    points = np.around(points).astype(np.int64)
    
    return points


# EVOLVE-BLOCK-END
