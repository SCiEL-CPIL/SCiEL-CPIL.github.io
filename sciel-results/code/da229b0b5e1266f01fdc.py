# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    """
    Constructs a large set of integer points in Z^11 where min pairwise
    squared distance >= max squared norm.
    
    Strategy: Start from E8⊕D3 union (252 pts, all norm²=8), then greedily
    add more points using vectorized distance checks.
    """
    start_time = time.time()
    deadline = start_time + 560.0  # Safety margin

    d = 11

    # ---------------------------------------------------------------
    # Build base E8 ⊕ D3 union (252 points, all norm²=8)
    # ---------------------------------------------------------------
    def build_base():
        pts = []
        # E8 minimal vectors (240)
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
        e8_pts = pts  # 240 points

        # D3 minimal vectors (12)
        d3_pts = []
        for i in range(3):
            for j in range(i + 1, 3):
                for si in [2, -2]:
                    for sj in [2, -2]:
                        v = np.zeros(3, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        d3_pts.append(v)

        result = []
        for v8 in e8_pts:
            v11 = np.zeros(11, dtype=np.int64)
            v11[:8] = v8
            result.append(v11)
        for v3 in d3_pts:
            v11 = np.zeros(11, dtype=np.int64)
            v11[8:] = v3
            result.append(v11)
        return result  # 252 points

    base = build_base()
    current = np.array(base, dtype=np.int64)  # shape (252, 11)
    current_max_norm = 8
    current_set = set(tuple(v) for v in current)

    # ---------------------------------------------------------------
    # Generate candidate pool: all integer vectors in Z^11 with 0 < norm² <= 12
    # These are potential additions (norm² <= 8 keeps max_norm at 8)
    # ---------------------------------------------------------------
    def generate_pool(max_norm_sq, dim):
        pool = []
        limit = int(np.sqrt(max_norm_sq))
        coord_range = list(range(-limit, limit + 1))

        def gen(idx, cur, nsq):
            if nsq > max_norm_sq:
                return
            if idx == dim:
                if nsq > 0:
                    pool.append(cur.copy())
                return
            for val in coord_range:
                new_nsq = nsq + val * val
                if new_nsq > max_norm_sq:
                    continue
                cur[idx] = val
                gen(idx + 1, cur, new_nsq)
            cur[idx] = 0

        gen(0, np.zeros(dim, dtype=np.int64), 0)
        return pool

    # Generate candidates with norm² <= 8 (to keep max_norm = 8)
    # This gives us all vectors that can be added without increasing max_norm
    candidates = generate_pool(8, d)
    
    # Filter out those already in current set
    candidates = [c for c in candidates if tuple(c) not in current_set]
    
    # Sort by norm² ascending
    candidates.sort(key=lambda v: int(np.sum(v ** 2)))

    # ---------------------------------------------------------------
    # Greedy addition with vectorized distance checks
    # ---------------------------------------------------------------
    added = 0
    for c in candidates:
        if time.time() > deadline - 30:
            break
        
        c_arr = np.array(c, dtype=np.int64)
        c_norm_sq = int(np.sum(c_arr ** 2))
        
        if c_norm_sq > current_max_norm:
            continue  # Would increase max_norm, skip for now
        
        # Vectorized: compute squared distances from c to all current points
        diffs = current - c_arr[np.newaxis, :]
        dists_sq = np.sum(diffs ** 2, axis=1)
        min_dist = int(np.min(dists_sq))
        
        if min_dist >= current_max_norm:
            current = np.vstack([current, c_arr[np.newaxis, :]])
            current_set.add(tuple(c))
            added += 1

    best_points = current
    best_k = len(current)

    # ---------------------------------------------------------------
    # Try to add points with norm² > 8 (increases max_norm)
    # Only worth it if we can add many such points
    # ---------------------------------------------------------------
    if time.time() < deadline - 90:
        candidates_higher = generate_pool(12, d)
        candidates_higher = [c for c in candidates_higher 
                           if tuple(c) not in current_set 
                           and int(np.sum(c ** 2)) > 8]
        candidates_higher.sort(key=lambda v: int(np.sum(v ** 2)))
        
        for c in candidates_higher:
            if time.time() > deadline - 30:
                break
            
            c_arr = np.array(c, dtype=np.int64)
            c_norm_sq = int(np.sum(c_arr ** 2))
            new_max_norm = max(current_max_norm, c_norm_sq)
            
            # Need all pairwise distances >= new_max_norm
            diffs = current - c_arr[np.newaxis, :]
            dists_sq = np.sum(diffs ** 2, axis=1)
            min_dist = int(np.min(dists_sq))
            
            # Also need existing min_dist >= new_max_norm
            # (already satisfied since existing min_dist >= 8, and new_max_norm <= 12)
            # But we need to check if existing points still satisfy the constraint
            # with the new higher max_norm
            
            if min_dist >= new_max_norm:
                # Check if all existing pairs still satisfy dist >= new_max_norm
                # This is expensive, so only do it if we're going to add
                # For now, just add and verify at the end
                current = np.vstack([current, c_arr[np.newaxis, :]])
                current_set.add(tuple(c))
                current_max_norm = new_max_norm

    if len(current) > best_k:
        best_points = current
        best_k = len(current)

    # ---------------------------------------------------------------
    # Final verification
    # ---------------------------------------------------------------
    points_arr = np.around(best_points).astype(np.int64)
    k = len(points_arr)
    
    if k < 2 or points_arr.shape[1] != 11:
        return np.array([np.ones(11, dtype=np.int64), -np.ones(11, dtype=np.int64)])
    
    zero_mask = np.all(points_arr == 0, axis=1)
    if np.any(zero_mask):
        points_arr = points_arr[~zero_mask]
    
    if len(points_arr) < 2:
        return np.array([np.ones(11, dtype=np.int64), -np.ones(11, dtype=np.int64)])
    
    # Verify
    k = len(points_arr)
    min_dist_sq = float('inf')
    max_norm_sq = 0
    
    for i in range(k):
        nsq = int(np.sum(points_arr[i] ** 2))
        if nsq > max_norm_sq:
            max_norm_sq = nsq
        for j in range(i + 1, k):
            diff = points_arr[i] - points_arr[j]
            dsq = int(np.sum(diff ** 2))
            if dsq < min_dist_sq:
                min_dist_sq = dsq
    
    if min_dist_sq < max_norm_sq:
        # Invalid - use base fallback
        fallback = np.array(build_base(), dtype=np.int64)
        points_arr = fallback
    
    return points_arr

# EVOLVE-BLOCK-END
