# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
import time


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates
    such that minimum squared pairwise distance >= maximum squared norm.
    Returns np.ndarray of shape (k, 11) with int64 entries.
    """
    start_time = time.time()
    deadline = start_time + 590  # safety margin before 630s kill

    d = 11

    # Base construction: all vectors with exactly 2 non-zero entries ±1
    # C(11,2) * 4 = 55 * 4 = 220 points
    points_list = []
    for i, j in combinations(range(d), 2):
        for si in (1, -1):
            for sj in (1, -1):
                p = np.zeros(d, dtype=np.int64)
                p[i] = si
                p[j] = sj
                points_list.append(p)

    points = np.array(points_list, dtype=np.int64)

    k = points.shape[0]

    # Quick validity check
    norms_sq = np.sum(points**2, axis=1)
    if k < 2 or np.any(norms_sq == 0):
        return np.array([[1,0,0,0,0,0,0,0,0,0,0],
                         [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)

    max_norm_sq = int(np.max(norms_sq))

    # Verify min pairwise dist >= max norm
    valid = True
    for i in range(k):
        for j in range(i + 1, k):
            diff = points[i] - points[j]
            dist_sq = int(np.dot(diff, diff))
            if dist_sq < max_norm_sq:
                valid = False
                break
        if not valid:
            break

    if not valid:
        return np.array([[1,0,0,0,0,0,0,0,0,0,0],
                         [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)

    best_points = points.copy()
    best_k = k

    # Try to add more points from a larger candidate pool
    # Generate candidate pool: all vectors with entries in {-1, 0, 1}
    # and exactly 2 or 3 non-zero entries
    # This gives C(11,2)*4 + C(11,3)*4*3 = 220 + 11*4*3*... let's compute
    # C(11,3) = 165, each with 3 non-zero entries each ±1: 165 * 8 = 1320
    # Total candidates: 220 + 1320 = 1540
    candidates = list(points_list)  # start with the 220
    for i, j, l in combinations(range(d), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sl in (1, -1):
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    p[l] = sl
                    candidates.append(p)

    candidate_arr = np.array(candidates, dtype=np.int64)
    n_cand = len(candidate_arr)

    # Greedy: try to add candidates that maintain the constraint
    # Current max norm is 2 (for weight-2 vectors) or will be 3 for weight-3
    # If we add weight-3 vectors, max_norm_sq becomes 3, so min dist must be >= 3

    # Strategy: try to build a set with max_norm_sq = 2 first (only weight-2)
    # Then try to extend with weight-3 vectors (max_norm_sq becomes 3)

    # For weight-2 only: min dist between two weight-2 vectors:
    # Same pair (i,j): e.g., (1,1) vs (1,-1) -> diff=(0,2), dist_sq=4 >= 2 ✓
    # Different pairs: e.g., (i,j) and (i,k): diff has entries at j,k -> dist_sq=2 >= 2 ✓
    # So all 220 weight-2 vectors work with max_norm_sq=2.

    # Now try adding weight-3 vectors with max_norm_sq=3:
    # A weight-3 vector v has norm_sq=3. Need dist(v, w) >= 3 for all w in set.
    # For w a weight-2 vector: dist_sq = |v-w|^2. 
    # v has 3 non-zero ±1, w has 2 non-zero ±1.
    # If w's support ⊂ v's support: dist_sq = 1 + 1 = 2 < 3 ✗
    # If w's support has 1 overlap: dist_sq depends on signs
    # If w's support disjoint from v's support: dist_sq = 3+2=5 >= 3 ✓

    # So we can only add weight-3 vectors whose support is disjoint from all weight-2 vectors in the set.
    # But our set contains ALL weight-2 vectors, so any weight-3 vector's support (size 3) 
    # contains a pair (i,j) which is a weight-2 vector in our set. 
    # dist would be 2 < 3. So we cannot add any weight-3 vectors to the full 220 set.

    # Alternative: use a subset of weight-2 vectors and add weight-3 vectors.
    # Or: try a different construction entirely.

    # Let's try: all weight-2 vectors (220) is our base. 
    # Can we add weight-1 vectors? A weight-1 vector e_i has norm_sq=1.
    # dist(e_i, w) for weight-2 w: if w has entry at i, dist_sq = 1+1=2 >= 1? 
    # max_norm_sq would still be 2 (from weight-2). dist(e_i, e_j) = 2 >= 2 ✓.
    # dist(e_i, w) where w has ±1 at i and ±1 at j: 
    #   if w[i]=1: e_i - w = (-1 at i, -1 at j) -> dist_sq=2
    #   if w[i]=-1: e_i - w = (2 at i, -1 at j) -> dist_sq=5
    # So dist >= 2 >= max_norm_sq=2 ✓. 
    # So we can add all 22 weight-1 vectors!

    # Add weight-1 vectors
    for i in range(d):
        for s in (1, -1):
            p = np.zeros(d, dtype=np.int64)
            p[i] = s
            candidates.append(p)

    # Rebuild candidate pool
    candidate_arr = np.array(candidates, dtype=np.int64)
    n_cand = len(candidate_arr)

    # Greedy addition: start with the 220 weight-2 points, try adding others
    # Check if time allows
    if time.time() < deadline - 120:
        # Try greedy: for each candidate not already in set, check if it can be added
        # Current set: 220 weight-2 vectors
        current_set = list(points_list)  # the original 220
        current_arr = np.array(current_set, dtype=np.int64)
        current_k = len(current_set)
        current_max_norm = 2  # all weight-2 have norm_sq=2

        # Try adding weight-1 vectors
        added_any = True
        while added_any and time.time() < deadline - 60:
            added_any = False
            for idx in range(n_cand):
                if time.time() > deadline - 60:
                    break
                cand = candidate_arr[idx]
                cnorm = int(np.dot(cand, cand))
                if cnorm == 0:
                    continue
                # Check if already in set
                already = False
                for existing in current_set:
                    if np.array_equal(cand, existing):
                        already = True
                        break
                if already:
                    continue
                new_max_norm = max(current_max_norm, cnorm)
                # Check dist to all current points
                ok = True
                for existing in current_set:
                    diff = cand - existing
                    dsq = int(np.dot(diff, diff))
                    if dsq < new_max_norm:
                        ok = False
                        break
                if ok:
                    current_set.append(cand)
                    current_arr = np.array(current_set, dtype=np.int64)
                    current_k += 1
                    if cnorm > current_max_norm:
                        current_max_norm = cnorm
                    added_any = True
                    if current_k > best_k:
                        best_k = current_k
                        best_points = np.array(current_set, dtype=np.int64)

    # Return the best found
    result = np.around(best_points).astype(np.int64)
    return result


# EVOLVE-BLOCK-END
