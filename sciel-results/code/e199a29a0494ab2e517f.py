# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional integer points such that the minimum
    squared distance between any two points is >= the maximum squared norm of any point.
    Uses a multi-stage strategy:
    1. High-Norm (8) Union Construction (E8 U D3)
    2. Low-Norm (4) Greedy Construction
    3. Medium-Norm (5-7) Greedy Construction
    Returns the best valid set found.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s kill
    d = 11

    def verify(points):
        if not points:
            return False, 0
        pts = np.array(points, dtype=np.int64)
        if pts.shape[1] != d:
            return False, 0
        if np.any(np.all(pts == 0, axis=1)):
            return False, 0
        
        k = len(pts)
        if k < 2:
            return False, 0
            
        max_norm_sq = 0
        for i in range(k):
            n = int(np.sum(pts[i]**2))
            if n > max_norm_sq:
                max_norm_sq = n
        
        min_dist_sq = float('inf')
        for i in range(k):
            for j in range(i+1, k):
                diff = pts[i] - pts[j]
                dist_sq = int(np.sum(diff**2))
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                if min_dist_sq < max_norm_sq:
                    return False, 0
                    
        return True, k

    def generate_candidates_up_to_norm(max_norm_sq):
        candidates = []
        coord_range = list(range(-3, 4)) # -3 to 3 is enough for norm <= 10
        
        def gen(idx, current, norm_sq):
            if norm_sq > max_norm_sq:
                return
            if idx == d:
                if norm_sq > 0:
                    candidates.append(current.copy())
                return
            # Pruning: max possible remaining norm
            # if norm_sq + (max_coord^2 * (d-idx)) < max_norm_sq? No, just check bound.
            for val in coord_range:
                new_norm = norm_sq + val * val
                if new_norm > max_norm_sq:
                    continue
                current[idx] = val
                gen(idx + 1, current, new_norm)
            current[idx] = 0

        gen(0, np.zeros(d, dtype=np.int64), 0)
        return candidates

    best_points = []
    best_k = 0

    # ---------------------------------------------------------------
    # Strategy 1: High Norm (8) - The existing Union Construction
    # ---------------------------------------------------------------
    def build_e8_minimal():
        pts = []
        for i in range(8):
            for j in range(i + 1, 8):
                for si in [2, -2]:
                    for sj in [2, -2]:
                        v = np.zeros(8, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        pts.append(v)
        for signs in itertools.product([1, -1], repeat=8):
            if sum(1 for s in signs if s == -1) % 2 == 0:
                pts.append(np.array(signs, dtype=np.int64))
        return pts

    def build_d3_minimal():
        pts = []
        for i in range(3):
            for j in range(i + 1, 3):
                for si in [2, -2]:
                    for sj in [2, -2]:
                        v = np.zeros(3, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        pts.append(v)
        return pts

    e8_pts = build_e8_minimal()
    d3_pts = build_d3_minimal()
    
    base_points = []
    for v8 in e8_pts:
        v11 = np.zeros(11, dtype=np.int64)
        v11[:8] = v8
        base_points.append(v11)
    for v3 in d3_pts:
        v11 = np.zeros(11, dtype=np.int64)
        v11[8:] = v3
        base_points.append(v11)
        
    valid, k = verify(base_points)
    if valid and k > best_k:
        best_points = base_points[:]
        best_k = k

    # ---------------------------------------------------------------
    # Strategy 2: Greedy on Low/Medium Norm Pools
    # ---------------------------------------------------------------
    for max_norm in [4, 5, 6, 7, 8]:
        if time.time() > deadline - 100:
            break
            
        candidates = generate_candidates_up_to_norm(max_norm)
        # Remove zero
        candidates = [p for p in candidates if np.any(p != 0)]
        
        # Greedy Selection with Random Restarts
        # We want to maximize size such that min_dist >= max_norm (which is <= max_norm)
        # Actually, the constraint is min_dist >= MAX_NORM_OF_SET.
        # If we restrict to vectors with norm <= max_norm, the max norm of the set will be <= max_norm.
        # So we need min_dist >= max_norm_of_set.
        # This is tricky because max_norm_of_set depends on the set.
        # Simplification: Enforce min_dist >= max_norm (the global cap).
        # This is a sufficient condition.
        
        best_in_strategy = []
        best_size = 0
        
        # Limit restarts based on time
        num_restarts = 50 if max_norm <= 5 else 10
        
        for _ in range(num_restarts):
            if time.time() > deadline - 20:
                break
                
            random.shuffle(candidates)
            current_set = []
            
            for p in candidates:
                # Check if p can be added
                # Constraint: dist(p, q) >= max_norm for all q in current_set
                # AND norm(p) <= max_norm
                # AND norm(q) <= max_norm (guaranteed by pool)
                # So we just need dist(p, q) >= max_norm.
                
                valid = True
                # Optimization: Check against current_set
                # If current_set is large, this is slow.
                # Vectorized check
                if current_set:
                    arr = np.array(current_set)
                    diffs = arr - p
                    dists_sq = np.sum(diffs**2, axis=1)
                    if np.min(dists_sq) < max_norm:
                        valid = False
                
                if valid:
                    current_set.append(p)
                    
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_in_strategy = current_set[:]
                
        # Verify the best found in this strategy
        # Note: The greedy enforced dist >= max_norm.
        # The actual max norm of the set might be smaller, so it's definitely valid.
        if best_size > best_k:
            # Double check validity
            valid, k = verify(best_in_strategy)
            if valid:
                best_points = best_in_strategy[:]
                best_k = k

    # ---------------------------------------------------------------
    # Final Fallback
    # ---------------------------------------------------------------
    if not best_points:
        best_points = [np.ones(11, dtype=np.int64), -np.ones(11, dtype=np.int64)]

    points_arr = np.array(best_points, dtype=np.int64)
    points_arr = np.around(points_arr).astype(np.int64)
    
    return points_arr

# EVOLVE-BLOCK-END
