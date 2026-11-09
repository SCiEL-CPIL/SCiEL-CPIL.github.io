# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that 
    their maximum squared norm is smaller than or equal to their minimum pairwise squared distance.
    
    Strategy:
    1. Try to find a large set of points with squared norm 3 using a greedy independent set heuristic.
    2. If the size is > 220, return that set.
    3. Otherwise, return the set of all 220 points with squared norm 2.
    """
    start_time = time.time()
    deadline = 600.0
    
    d = 11
    
    # --- Attempt R^2 = 3 ---
    # Generate all vectors with squared norm 3
    # These are vectors with exactly 3 non-zero coordinates, each +/-1
    candidates = []
    indices = range(d)
    
    for i in indices:
        for j in range(i + 1, d):
            for k in range(j + 1, d):
                for si in [-1, 1]:
                    for sj in [-1, 1]:
                        for sk in [-1, 1]:
                            p = [0] * d
                            p[i] = si
                            p[j] = sj
                            p[k] = sk
                            candidates.append(p)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # Check time
    if time.time() > deadline - 30:
        return _get_norm2_points()

    # Build conflict graph? Or just greedy check.
    # Greedy: Pick a random order, add if no conflict with previously added.
    # Conflict: dist_sq < 3  <=>  dot_product > 1.5  <=> dot_product >= 2.
    
    best_set = []
    best_size = 0
    
    # Try a few random permutations to improve greedy result
    # But 1320 nodes, 50 iterations might be slow in Python loop.
    # Let's do 10 iterations.
    
    # Precompute dot products? 1320x1320 matrix is ~1.7M entries.
    # 1.7M * 8 bytes = 13.6 MB. Fits in memory.
    # Calculation: candidates @ candidates.T
    # This is fast in NumPy.
    
    try:
        dot_prods = candidates @ candidates.T
        # We need dot_prods[i, j] < 2 for i != j to be compatible.
        # Actually, we need dist_sq >= 3.
        # dist_sq = 3 + 3 - 2 dot = 6 - 2 dot.
        # 6 - 2 dot >= 3 => 2 dot <= 3 => dot <= 1.5 => dot <= 1.
        # So compatible if dot <= 1.
        
        # Mask of conflicts: dot >= 2
        conflict_mask = dot_prods >= 2
        
        # Diagonal is 3 (dot of self), which is >= 2, but we don't pick same node twice.
        
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
                
            order = np.random.permutation(n_cand)
            selected = []
            selected_indices = set()
            
            for idx in order:
                # Check if idx conflicts with any selected
                # We can maintain a boolean array of "blocked" nodes.
                # When we select idx, we block all j where conflict_mask[idx, j] is True.
                
                # Efficient way:
                # Keep a set of available indices.
                # Initially all.
                # When selecting idx, remove idx and all its neighbors from available.
                
                # Let's use a list of booleans for availability
                # But resetting 1320 bools 20 times is cheap.
                
                available = np.ones(n_cand, dtype=bool)
                
                # Re-do the loop with available array
                selected = []
                for idx in order:
                    if available[idx]:
                        selected.append(idx)
                        # Block idx and its neighbors
                        available[idx] = False
                        # Get neighbors
                        neighbors = np.where(conflict_mask[idx])[0]
                        available[neighbors] = False
                
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_set = selected
                    
    except Exception as e:
        # Fallback to norm 2 if something goes wrong
        return _get_norm2_points()

    if best_size > 220:
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
            return points

    # Fallback to R^2 = 2
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
