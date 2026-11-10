# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates
    such that min squared pairwise distance >= max squared norm.
    Returns np.ndarray of shape (k, 11) with int64 entries.
    """
    d = 11
    deadline = time.time() + 620  # safety margin before 630s kill

    # Strategy: use vectors with exactly 2 non-zero coordinates, each ±1.
    # There are C(11,2) * 4 = 55 * 4 = 220 such vectors.
    # Each has squared norm = 2.
    # For any two distinct such vectors:
    #   - If they share no coordinates: dist^2 = 2 + 2 = 4 >= 2 ✓
    #   - If they share one coordinate with same sign: dist^2 = 0 + 1 + 1 = 2 >= 2 ✓
    #   - If they share one coordinate with opposite sign: dist^2 = 0 + 2 + 2 = 4 >= 2 ✓
    #   - If they share both coordinates (i.e., same pair):
    #       Same signs: dist^2 = 0 (not distinct)
    #       One sign flip: dist^2 = 2 >= 2 ✓
    #       Both sign flip: dist^2 = 4 >= 2 ✓
    # So all 220 vectors are valid!

    points = []
    for i, j in itertools.combinations(range(d), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(d, dtype=np.int64)
                v[i] = si
                v[j] = sj
                points.append(v)

    points = np.array(points, dtype=np.int64)

    # Verify validity
    k = len(points)
    if k < 2:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)

    # Check no zero vector
    norms = np.sum(points ** 2, axis=1)
    if np.any(norms == 0):
        # Remove zero vectors
        mask = norms > 0
        points = points[mask]
        k = len(points)

    # Check min pairwise dist >= max norm
    max_norm = int(np.max(norms))
    min_dist_sq = float('inf')
    for a in range(k):
        for b in range(a + 1, k):
            diff = points[a] - points[b]
            dist_sq = int(np.sum(diff ** 2))
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    if min_dist_sq >= max_norm:
        return points

    # Fallback: return the 2-point solution
    return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)


# EVOLVE-BLOCK-END
