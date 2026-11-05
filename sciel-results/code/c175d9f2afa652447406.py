def construct():
    import time
    import random
    import numpy as np
    
    deadline = time.time() + 290
    
    N = 64
    all_points = [(x, y) for y in range(N) for x in range(N)]
    all_points_arr = np.array(all_points, dtype=np.int32)
    point_to_idx = {p: i for i, p in enumerate(all_points)}
    
    def sq_dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx*dx + dy*dy

    def is_isosceles(p1, p2, p3):
        d12 = sq_dist(p1, p2)
        d13 = sq_dist(p1, p3)
        d23 = sq_dist(p2, p3)
        return d12 == d13 or d12 == d23 or d13 == d23

    def can_add_fast(p, S, dist_sets):
        dists_from_p = []
        for q in S:
            d = sq_dist(p, q)
            if d in dists_from_p:
                return False
            dists_from_p.append(d)
            if d in dist_sets[q]:
                return False
        return True

    def add_point(S, dist_sets, p):
        S.append(p)
        dists = []
        for q in S[:-1]:
            d = sq_dist(p, q)
            dists.append(d)
            dist_sets[q].add(d)
        dist_sets[p] = set(dists)

    def remove_point(S, dist_sets, p):
        S.remove(p)
        for q in S:
            d = sq_dist(p, q)
            if d in dist_sets[q]:
                dist_sets[q].discard(d)
        del dist_sets[p]

    def greedy(order):
        S = []
        dist_sets = {}
        for p in order:
            if time.time() > deadline:
                break
            if p in dist_sets:
                continue
            if can_add_fast(p, S, dist_sets):
                add_point(S, dist_sets, p)
        return S, dist_sets

    def local_search(S, dist_sets):
        """Swap-based local search: try removing each point and adding the best replacement."""
        current_S = list(S)
        current_dist = {k: v.copy() for k, v in dist_sets.items()}
        best_S = list(current_S)
        best_dist = {k: v.copy() for k, v in current_dist.items()}
        best_size = len(current_S)
        
        rng = random.Random(12345)
        
        # Precompute all pair squared distances for vectorized candidate filtering
        n_pts = len(all_points)
        # dist_matrix[i,j] = squared distance between all_points[i] and all_points[j]
        coords = all_points_arr
        dx = coords[:, 0:1] - coords[1:2, 0]  # not quite right, let's do it properly
        # Actually: for each pair (i,j), dist = (x_i - x_j)^2 + (y_i - y_j)^2
        xs = coords[:, 0].astype(np.int64)
        ys = coords[:, 1].astype(np.int64)
        dist_matrix = (xs[:, None] - xs[None, :])**2 + (ys[:, None] - ys[None, :])**2
        
        while time.time() < deadline - 10:
            improved = False
            indices = list(range(len(current_S)))
            rng.shuffle(indices)
            
            for idx in indices:
                if time.time() > deadline - 10:
                    break
                if len(current_S) <= best_size:
                    break
                
                p_remove = current_S[idx]
                
                remove_point(current_S, current_dist, p_remove)
                
                # Get indices of points in current_S
                S_indices = [point_to_idx[q] for q in current_S]
                
                # Vectorized check: for each candidate not in S, check if it can be added
                # A candidate c can be added if:
                # 1. c is not already in S
                # 2. For all q in S: dist(c,q) is not in dist_sets[q]
                # 3. All dist(c,q) for q in S are distinct
                
                # Build a set of forbidden distances per S point
                # dist_sets[q] is a set of distances from q to other points in S
                # For candidate c, dist(c, q) must not be in dist_sets[q] for any q in S
                
                # Vectorized approach:
                # For each candidate c (not in S), compute distances to all S points
                # Check if any distance is in the corresponding dist_set
                # Also check if distances are all distinct
                
                candidate_indices = set(range(n_pts)) - set(S_indices)
                if not candidate_indices:
                    add_point(current_S, current_dist, p_remove)
                    continue
                
                cand_arr = np.array(list(candidate_indices), dtype=np.int32)
                
                # Distances from candidates to S points: shape (len(cand_arr), len(S))
                cand_coords = all_points_arr[cand_arr]  # (C, 2)
                S_coords = all_points_arr[S_indices]  # (S, 2)
                
                dx_c = cand_coords[:, 0:1] - S_coords[0:1, 0]  # (C, 1)
                dy_c = cand_coords[:, 1:2] - S_coords[0:1, 1]  # (C, 1)
                # This is wrong, need full outer product
                dx_full = cand_coords[:, 0:1] - S_coords[None, :, 0]  # (C, S)
                dy_full = cand_coords[:, 1:2] - S_coords[None, :, 1]  # (C, S)
                dists_c = dx_full**2 + dy_full**2  # (C, S)
                
                # Check condition 3: all distances in each row are distinct
                # Sort each row and check for duplicates
                sorted_dists = np.sort(dists_c, axis=1)
                has_dup = np.any(sorted_dists[:, 1:] == sorted_dists[:, :-1], axis=1)
                
                # Check condition 2: for each candidate c and each S point q_i, 
                # dist(c, q_i) should not be in dist_sets[q_i]
                # dist_sets is a dict: point -> set of distances
                # We need to check dists_c[i, j] not in dist_sets[current_S[j]]
                
                # Build a 2D array of forbidden distances: forbidden[i, j] = 1 if dists_c[i,j] in dist_sets[current_S[j]]
                # This is hard to vectorize directly since dist_sets values are sets
                # But we can check: for each S point j, create a set, then check membership
                
                # Alternative: for each S point, the forbidden distances are in dist_sets
                # Let's do a loop over S points but vectorize over candidates
                valid_mask = ~has_dup.copy()
                
                for j, q in enumerate(current_S):
                    forbidden_set = dist_sets[q]
                    # Check which candidates have dists_c[:, j] in forbidden_set
                    d_col = dists_c[:, j]
                    # Vectorized membership test: create a set and check
                    # Since distances can be large, use a set
                    in_forbidden = np.array([d in forbidden_set for d in d_col])
                    valid_mask &= ~in_forbidden
                
                if not np.any(valid_mask):
                    add_point(current_S, current_dist, p_remove)
                    continue
                
                # Pick a random valid candidate
                valid_indices = np.where(valid_mask)[0]
                chosen = valid_indices[rng.randint(0, len(valid_indices) - 1)]
                p_add = all_points[cand_arr[chosen]]
                
                add_point(current_S, current_dist, p_add)
                
                if len(current_S) > best_size:
                    best_S = list(current_S)
                    best_dist = {k: v.copy() for k, v in current_dist.items()}
                    best_size = len(current_S)
                    improved = True
            
            if not improved:
                break
        
        return best_S, best_dist

    best_S = []
    best_dist = {}
    
    rng = random.Random(42)
    
    for i in range(15):
        if time.time() > deadline - 20:
            break
        order = all_points.copy()
        rng.shuffle(order)
        S, dist_sets = greedy(order)
        if len(S) > len(best_S):
            best_S = S
            best_dist = dist_sets
    
    if len(best_S) > 0 and time.time() < deadline - 15:
        best_S, best_dist = local_search(best_S, best_dist)
    
    valid_S = list(best_S)
    changed = True
    while changed and time.time() < deadline - 1:
        changed = False
        n = len(valid_S)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if is_isosceles(valid_S[i], valid_S[j], valid_S[k]):
                        valid_S.pop(k)
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break

    return valid_S
