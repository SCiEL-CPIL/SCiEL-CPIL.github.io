"""No-isosceles set construction with vectorized validation and local search."""
import time
import random
import numpy as np

N = 64
TIME_LIMIT = 280.0  # Safety margin before 330s kill

def construct():
    start_time = time.time()
    
    # Precompute all grid points
    all_points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int64)
    
    def is_valid_set(pts):
        """Check if a set of points has no isosceles triangles."""
        if len(pts) < 3:
            return True
        pts = np.array(pts)
        n = len(pts)
        # Compute all pairwise squared distances
        # d[i,j] = squared distance between pts[i] and pts[j]
        dx = pts[:, 0:1] - pts[1:, 0].T
        dy = pts[:, 1:2] - pts[1:, 1].T
        d = dx**2 + dy**2
        # For each triple (i, j, k), check if any two distances are equal
        # We need to check all triples. For efficiency, we check:
        # For each pair (i,j), and each k != i,j, check if d(i,k)==d(j,k) or d(i,j)==d(i,k) or d(i,j)==d(j,k)
        # But this is O(n^3). Let's use a smarter approach.
        
        # Alternative: for each point k, compute distances to all other points.
        # If any two distances from k to other points are equal, we have an isosceles triangle
        # with k as the apex. But we also need to check if k is the base vertex.
        # Actually, an isosceles triangle with vertices a,b,c has either:
        # - d(a,b) == d(a,c) [a is apex]
        # - d(a,b) == d(b,c) [b is apex]  
        # - d(a,c) == d(b,c) [c is apex]
        
        # For each point k, let dists_k = [d(k, j) for j != k]
        # If any value appears more than once in dists_k, then there exist i, j with d(k,i) == d(k,j),
        # which means triangle (k, i, j) is isosceles with k as apex.
        # But we also need to check cases where k is NOT the apex.
        # Actually, if we check for every point k whether any two distances from k are equal,
        # that catches all isosceles triangles because in any isosceles triangle, the apex is one of the three points.
        
        valid = True
        for k in range(n):
            # Distances from k to all other points
            if k == 0:
                dists = d[0, :]  # distances from point 0 to points 1..n-1
            elif k == n - 1:
                dists = d[k-1, k-1]  # This is wrong, let me restructure
            else:
                dists = np.concatenate([d[0:k-1, k-1], d[k, k-1:]])
            # Check for duplicates
            if len(dists) >= 2:
                if len(np.unique(dists)) < len(dists):
                    return False
        return True
    
    def can_add_point(new_pt, existing_pts):
        """Check if adding new_pt to existing_pts creates no isosceles triangles."""
        if len(existing_pts) < 2:
            return True
        existing_pts = np.array(existing_pts)
        n = len(existing_pts)
        new_pt = np.array(new_pt)
        
        # Distances from new_pt to all existing points
        dx = new_pt[0] - existing_pts[:, 0]
        dy = new_pt[1] - existing_pts[:, 1]
        dists_new = dx**2 + dy**2
        
        # Check if new_pt is apex: any two distances from new_pt equal?
        if len(dists_new) >= 2:
            if len(np.unique(dists_new)) < len(dists_new):
                return False
        
        # Check if any existing point is apex: for each existing point i,
        # check if d(i, new_pt) == d(i, j) for some j != i, new_pt
        for i in range(n):
            # d(i, new_pt) = dists_new[i]
            # d(i, j) for j != i: need to check if any equals dists_new[i]
            # d(i, j) = (xi-xj)^2 + (yi-yj)^2
            dx_ij = existing_pts[i, 0] - existing_pts[:, 0]
            dy_ij = existing_pts[i, 1] - existing_pts[:, 1]
            dists_ij = dx_ij**2 + dy_ij**2
            # Exclude i itself (d(i,i) = 0)
            dists_ij[i] = -1
            # Check if dists_new[i] appears in dists_ij (excluding the -1 we set)
            if dists_new[i] > 0:  # new_pt != existing_pts[i]
                if dists_new[i] in dists_ij:
                    return False
        
        return True
    
    # Greedy construction
    def greedy_construct(seed=0):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(all_points))
        selected = []
        selected_set = set()
        
        for idx in order:
            pt = tuple(all_points[idx])
            if pt in selected_set:
                continue
            if can_add_point(pt, selected):
                selected.append(pt)
                selected_set.add(pt)
        
        return selected
    
    # Local search: try to improve by swapping points
    def local_search(initial_set, max_iters=500):
        if not initial_set:
            return initial_set
        
        current_set = list(initial_set)
        current_set_set = set(current_set)
        best_set = list(current_set)
        best_len = len(current_set)
        
        for iteration in range(max_iters):
            if time.time() - start_time > TIME_LIMIT:
                break
            
            improved = False
            # Try removing a random point and adding a different one
            if len(current_set) > 0:
                rem_idx = random.randint(0, len(current_set) - 1)
                rem_pt = current_set[rem_idx]
                temp_set = [p for i, p in enumerate(current_set) if i != rem_idx]
                temp_set_set = set(temp_set)
                
                # Try to find a point to add
                candidates = [p for p in all_points if tuple(p) not in temp_set_set]
                random.shuffle(candidates)
                
                for cand in candidates:
                    cand_tuple = tuple(cand)
                    if can_add_point(cand_tuple, temp_set):
                        new_set = temp_set + [cand_tuple]
                        new_set_set = set(new_set)
                        if len(new_set) > best_len:
                            best_len = len(new_set)
                            best_set = list(new_set)
                        current_set = new_set
                        current_set_set = new_set_set
                        improved = True
                        break
            if not improved and random.random() < 0.1:
                # Occasionally try pure addition
                candidates = [p for p in all_points if tuple(p) not in current_set_set]
                random.shuffle(candidates)
                for cand in candidates:
                    cand_tuple = tuple(cand)
                    if can_add_point(cand_tuple, current_set):
                        current_set.append(cand_tuple)
                        current_set_set.add(cand_tuple)
                        if len(current_set) > best_len:
                            best_len = len(current_set)
                            best_set = list(current_set)
                        break
        
        return best_set
    
    # Run multiple greedy constructions with different seeds
    best_result = []
    num_trials = 20
    
    for seed in range(num_trials):
        if time.time() - start_time > TIME_LIMIT:
            break
        
        greedy_set = greedy_construct(seed)
        
        # Local search to improve
        improved_set = local_search(greedy_set, max_iters=100)
        
        if len(improved_set) > len(best_result):
            best_result = improved_set
    
    return best_result
