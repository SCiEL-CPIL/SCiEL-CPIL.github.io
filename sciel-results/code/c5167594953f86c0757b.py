# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations, product


def kissing_number11() -> np.ndarray:
    """
    Constructs a set of eleven-dimensional integer points satisfying the kissing
    number condition: min pairwise squared distance >= max squared norm.
    
    Strategy: Try to build a larger set than the 220-point 2-nonzero solution.
    First try 3-nonzero vectors with greedy selection, fall back to 2-nonzero.
    """
    d = 11
    deadline = time.time() + 550  # safety margin before 600s limit
    
    # Baseline: 220 points with exactly 2 non-zero entries
    def build_2nonzero():
        points_list = []
        for i, j in combinations(range(d), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    points_list.append(p)
        return np.array(points_list, dtype=np.int64)
    
    best_points = build_2nonzero()
    best_k = len(best_points)
    
    # Try 3-nonzero vectors: all have norm^2 = 3
    # Constraint: for any two selected, dist^2 >= 3
    # Two 3-nonzero vectors sharing 2 indices with both signs matching give dist^2 = 2 < 3 (bad)
    # So we need to select a subset where no two share 2+ indices with matching signs on all shared.
    
    # Generate all 3-nonzero vectors
    all_3nz = []
    for i, j, k in combinations(range(d), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    p[k] = sk
                    all_3nz.append(p)
    
    all_3nz_arr = np.array(all_3nz, dtype=np.int64)
    n_3nz = len(all_3nz_arr)
    
    # Greedy selection: add vectors in order, keep only if compatible with all already selected
    # Compatibility: dist^2 >= 3 for all pairs
    selected = []
    selected_arr = []
    
    for idx in range(n_3nz):
        if time.time() > deadline:
            break
        p = all_3nz_arr[idx]
        compatible = True
        if selected_arr:
            # Compute squared distances to all selected
            diffs = selected_arr - p
            dists_sq = np.sum(diffs * diffs, axis=1)
            if np.min(dists_sq) < 3:
                compatible = False
        if compatible:
            selected_arr.append(p)
    
    if selected_arr:
        candidate = np.array(selected_arr, dtype=np.int64)
        if len(candidate) > best_k:
            best_points = candidate
            best_k = len(candidate)
    
    best_points = np.around(best_points).astype(np.int64)
    return best_points


# EVOLVE-BLOCK-END
