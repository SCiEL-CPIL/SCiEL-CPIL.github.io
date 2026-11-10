import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    """
    Returns a set of integer points in 11D where min pairwise dist sq >= max norm sq.
    Uses greedy independent set on norm-4 vectors with dot product constraint.
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 600s limit
    
    d = 11
    
    # Generate all integer vectors with squared norm 4
    candidates = []
    
    # Type 1: single ±2 (22 vectors)
    for i in range(d):
        for sign in [1, -1]:
            v = [0] * d
            v[i] = 2 * sign
            candidates.append(v)
    
    # Type 2: four coordinates ±1 (C(11,4)*16 = 5280 vectors)
    for idx in combinations(range(d), 4):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    for s4 in [1, -1]:
                        v = [0] * d
                        v[idx[0]] = s1
                        v[idx[1]] = s2
                        v[idx[2]] = s3
                        v[idx[3]] = s4
                        candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # Compute dot product matrix using int32 for memory
    cand_int32 = candidates.astype(np.int32)
    dot_matrix = cand_int32 @ cand_int32.T
    
    # Conflict: dot > 2 for distinct pairs
    conflict = (dot_matrix > 2)
    np.fill_diagonal(conflict, False)
    
    # Greedy independent set with multiple strategies, keep best
    best_selected = []
    
    # Strategy 1: min-degree greedy
    remaining = np.ones(n_cand, dtype=bool)
    degrees = conflict.sum(axis=1).copy()
    selected = []
    
    while remaining.any():
        if time.time() > deadline - 120:
            break
        rem_indices = np.where(remaining)[0]
        if len(rem_indices) == 0:
            break
        rem_degrees = degrees[rem_indices]
        min_idx = rem_indices[np.argmin(rem_degrees)]
        selected.append(min_idx)
        remaining[min_idx] = False
        neighbors = np.where(conflict[min_idx, :])[0]
        remaining[neighbors] = False
        for nb in neighbors:
            if remaining[nb]:
                degrees[nb] -= 1
    
    if len(selected) > len(best_selected):
        best_selected = selected[:]
    
    # Strategy 2: random order greedy (a few random restarts)
    rng = np.random.RandomState(42)
    for restart in range(5):
        if time.time() > deadline - 120:
            break
        
        order = rng.permutation(n_cand)
        remaining = np.ones(n_cand, dtype=bool)
        selected = []
        
        for idx in order:
            if not remaining[idx]:
                continue
            selected.append(idx)
            remaining[idx] = False
            neighbors = np.where(conflict[idx, :])[0]
            remaining[neighbors] = False
        
        if len(selected) > len(best_selected):
            best_selected = selected[:]
    
    # Strategy 3: max-neighborhood greedy (pick vertex that conflicts with most already-excluded)
    if time.time() < deadline - 120:
        remaining = np.ones(n_cand, dtype=bool)
        selected = []
        excluded_count = np.zeros(n_cand, dtype=np.int32)
        
        while remaining.any():
            if time.time() > deadline - 120:
                break
            rem_indices = np.where(remaining)[0]
            if len(rem_indices) == 0:
                break
            # Pick vertex with max excluded_count (most conflicts with already picked)
            rem_excl = excluded_count[rem_indices]
            pick_idx = rem_indices[np.argmax(rem_excl)]
            selected.append(pick_idx)
            remaining[pick_idx] = False
            neighbors = np.where(conflict[pick_idx, :])[0]
            for nb in neighbors:
                if remaining[nb]:
                    excluded_count[nb] += 1
            remaining[neighbors] = False
        
        if len(selected) > len(best_selected):
            best_selected = selected[:]
    
    result = candidates[best_selected]
    
    # Verify the solution
    if len(result) < 2:
        # Fallback to norm-2 solution
        indices = [(i, j) for i in range(d) for j in range(i + 1, d)]
        points = np.zeros((len(indices) * 4, d), dtype=np.int64)
        idx = 0
        for i, j in indices:
            points[idx, i] = 1
            points[idx, j] = 1
            idx += 1
            points[idx, i] = 1
            points[idx, j] = -1
            idx += 1
            points[idx, i] = -1
            points[idx, j] = 1
            idx += 1
            points[idx, i] = -1
            points[idx, j] = -1
            idx += 1
        return points
    
    # Verify using vectorized operations
    norms_sq = np.sum(result ** 2, axis=1)
    max_norm = int(np.max(norms_sq))
    
    # Check pairwise distances
    min_dist = float('inf')
    n_sel = len(result)
    for i in range(n_sel):
        if time.time() > deadline - 30:
            break
        diffs = result[i+1:] - result[i]
        dists = np.sum(diffs ** 2, axis=1)
        if len(dists) > 0:
            min_d = int(np.min(dists))
            if min_d < min_dist:
                min_dist = min_d
    
    if min_dist < max_norm:
        # Invalid, fallback to norm-2 solution
        indices = [(i, j) for i in range(d) for j in range(i + 1, d)]
        points = np.zeros((len(indices) * 4, d), dtype=np.int64)
        idx = 0
        for i, j in indices:
            points[idx, i] = 1
            points[idx, j] = 1
            idx += 1
            points[idx, i] = 1
            points[idx, j] = -1
            idx += 1
            points[idx, i] = -1
            points[idx, j] = 1
            idx += 1
            points[idx, i] = -1
            points[idx, j] = -1
            idx += 1
        return points
    
    return np.around(result).astype(np.int64)
