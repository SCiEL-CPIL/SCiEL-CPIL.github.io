# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    """
    Finds a large set of integer points in R^11 such that
    min pairwise squared distance >= max squared norm.
    
    Strategy: Mixed candidate pool (norm^2 = 2 and norm^2 = 4 vectors),
    degree-ordered greedy + local improvement, targeting min_dist_sq >= 4.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 600s limit
    
    d = 11
    candidates = []
    
    # Generate Norm Squared 2 vectors: two coords +/-1
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * d
                    v[i] = s1
                    v[j] = s2
                    candidates.append(v)
    
    # Generate Norm Squared 4 vectors:
    # Type A: one coord +/-2
    for i in range(d):
        for s in [2, -2]:
            v = [0] * d
            v[i] = s
            candidates.append(v)
    
    # Type B: four coords +/-1
    for combo in itertools.combinations(range(d), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * d
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # Check time
    if time.time() > deadline - 10:
        return _get_norm2_points()
    
    # Compute conflict matrix: conflict if dist_sq < 4
    # dist_sq = ||a||^2 + ||b||^2 - 2*a.b
    # For norm^2 2 and 2: dist_sq = 4 - 2*dot, conflict if dot > 0 => dot >= 1
    # For norm^2 2 and 4: dist_sq = 6 - 2*dot, conflict if dot > 1 => dot >= 2
    # For norm^2 4 and 4: dist_sq = 8 - 2*dot, conflict if dot > 2 => dot >= 3
    
    dot_prods = candidates @ candidates.T
    norms_sq = np.sum(candidates**2, axis=1)
    
    # Build conflict mask
    conflict_mask = np.zeros((n_cand, n_cand), dtype=bool)
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            dist_sq = norms_sq[i] + norms_sq[j] - 2 * dot_prods[i, j]
            if dist_sq < 4:
                conflict_mask[i, j] = True
                conflict_mask[j, i] = True
    
    if time.time() > deadline - 5:
        return _get_norm2_points()
    
    # Degree of each vertex
    degrees = np.sum(conflict_mask, axis=1)
    
    best_set = []
    best_size = 0
    
    # Multiple strategies:
    # 1. Degree-ordered greedy (ascending degree)
    # 2. Random restarts with local improvement
    
    # Strategy 1: Degree-ordered greedy
    order = np.argsort(degues) if False else np.argsort(degrees)
    selected = []
    available = np.ones(n_cand, dtype=bool)
    
    for idx in order:
        if available[idx]:
            selected.append(idx)
            available[idx] = False
            neighbors = np.where(conflict_mask[idx])[0]
            available[neighbors] = False
    
    if len(selected) > best_size:
        best_size = len(selected)
        best_set = selected[:]
    
    # Strategy 2: Random restarts with local improvement
    n_restarts = 0
    while time.time() < deadline - 20:
        n_restarts += 1
        order = np.random.permutation(n_cand)
        selected = []
        available = np.ones(n_cand, dtype=bool)
        
        for idx in order:
            if available[idx]:
                selected.append(idx)
                available[idx] = False
                neighbors = np.where(conflict_mask[idx])[0]
                available[neighbors] = False
        
        # Local improvement: try to swap
        # For each selected vertex, try replacing it with an unselected non-conflicting vertex
        improved = True
        max_improvements = 10
        while improved and max_improvements > 0:
            improved = False
            max_improvements -= 1
            selected_set = set(selected)
            for pos in range(len(selected)):
                if time.time() > deadline - 10:
                    break
                curr = selected[pos]
                # Find unselected vertices that don't conflict with (selected - {curr})
                others = [s for i, s in enumerate(selected) if i != pos]
                for cand_idx in range(n_cand):
                    if cand_idx in selected_set:
                        continue
                    # Check if cand_idx conflicts with any in others
                    conflict = False
                    for o in others:
                        if conflict_mask[cand_idx, o]:
                            conflict = True
                            break
                    if not conflict:
                        # Swap
                        selected[pos] = cand_idx
                        selected_set.remove(curr)
                        selected_set.add(cand_idx)
                        improved = True
                        break
                if time.time() > deadline - 10:
                    break
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
            if best_size > 300:
                # Good result, continue searching but with time check
                pass
    
    # Build result
    if best_size >= 2:
        points = candidates[best_set]
        # Verify
        norms_sq = np.sum(points**2, axis=1)
        max_norm_sq = np.max(norms_sq)
        
        min_dist_sq = float('inf')
        n_points = len(points)
        for i in range(n_points):
            for j in range(i + 1, n_points):
                diff = points[i] - points[j]
                dist_sq = np.sum(diff**2)
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
        
        if min_dist_sq >= max_norm_sq:
            return np.array(points, dtype=np.int64)
    
    # Fallback to norm^2 = 2 points
    return _get_norm2_points()

def _get_norm2_points():
    d = 11
    points = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    p = [0] * d
                    p[i] = si
                    p[j] = sj
                    points.append(p)
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
