import numpy as np
import itertools
import time
import random

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    num_points_total = len(grid)
    deadline = time.time() + 580  # Safety margin

    # Pre-compute squared norms for each point
    sq_norms = np.sum(grid ** 2, axis=1)

    def compute_sphere(p1, p2, p3, p4):
        """
        Compute the sphere passing through 4 points.
        Returns (cx, cy, cz, r_sq) or None if coplanar.
        """
        # Matrix A: [x, y, z, 1]
        # We want to solve for the sphere equation:
        # x^2 + y^2 + z^2 + Ax + By + Cz + D = 0
        # This is a linear system in A, B, C, D.
        # For point p_i: |p_i|^2 + A x_i + B y_i + C z_i + D = 0
        # A x_i + B y_i + C z_i + D = -|p_i|^2
        
        A = np.array([
            [p1[0], p1[1], p1[2], 1],
            [p2[0], p2[1], p2[2], 1],
            [p3[0], p3[1], p3[2], 1],
            [p4[0], p4[1], p4[2], 1]
        ], dtype=np.float64)
        
        b = np.array([-sq_norms[p1_idx], -sq_norms[p2_idx], -sq_norms[p3_idx], -sq_norms[p4_idx]], dtype=np.float64)
        
        # Check if points are coplanar (det(A) == 0)
        det_A = np.linalg.det(A)
        if abs(det_A) < 1e-9:
            return None # Coplanar, no unique sphere
            
        try:
            coeffs = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None
            
        A_coef, B_coef, C_coef, D_coef = coeffs
        
        # Center: (-A/2, -B/2, -C/2)
        cx = -A_coef / 2.0
        cy = -B_coef / 2.0
        cz = -C_coef / 2.0
        
        # Radius squared: cx^2 + cy^2 + cz^2 - D
        r_sq = cx*cx + cy*cy + cz*cz - D_coef
        
        return (cx, cy, cz, r_sq)

    # We need indices for sq_norms
    # The function compute_sphere takes point indices or coordinates?
    # It's better to pass indices to use pre-computed sq_norms.
    
    def compute_sphere_idx(i1, i2, i3, i4):
        p1 = grid[i1]
        p2 = grid[i2]
        p3 = grid[i3]
        p4 = grid[i4]
        
        A = np.array([
            [p1[0], p1[1], p1[2], 1],
            [p2[0], p2[1], p2[2], 1],
            [p3[0], p3[1], p3[2], 1],
            [p4[0], p4[1], p4[2], 1]
        ], dtype=np.float64)
        
        b = np.array([-sq_norms[i1], -sq_norms[i2], -sq_norms[i3], -sq_norms[i4]], dtype=np.float64)
        
        det_A = np.linalg.det(A)
        if abs(det_A) < 1e-9:
            return None
            
        try:
            coeffs = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None
            
        A_coef, B_coef, C_coef, D_coef = coeffs
        cx = -A_coef / 2.0
        cy = -B_coef / 2.0
        cz = -C_coef / 2.0
        r_sq = cx*cx + cy*cy + cz*cz - D_coef
        
        return (cx, cy, cz, r_sq)

    def is_on_sphere(p_idx, sphere):
        if sphere is None:
            return False
        cx, cy, cz, r_sq = sphere
        p = grid[p_idx]
        dist_sq = (p[0]-cx)**2 + (p[1]-cy)**2 + (p[2]-cz)**2
        return abs(dist_sq - r_sq) < 1e-9

    def greedy_build(order):
        curr_indices = []
        active_spheres = [] # List of (sphere_params, indices_tuple)
        
        for idx in order:
            if time.time() > deadline:
                break
                
            # Check if idx is on any active sphere
            valid = True
            for sphere, _ in active_spheres:
                if is_on_sphere(idx, sphere):
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Add idx
            curr_indices.append(idx)
            
            # Generate new spheres involving idx and 3 other points in curr_indices
            # The other 3 points must be from curr_indices[:-1]
            if len(curr_indices) >= 4:
                others = curr_indices[:-1]
                for combo in itertools.combinations(others, 3):
                    sphere_params = compute_sphere_idx(idx, combo[0], combo[1], combo[2])
                    if sphere_params is not None:
                        active_spheres.append((sphere_params, (idx, combo[0], combo[1], combo[2])))
                        
        return np.array(curr_indices, dtype=np.int64), active_spheres

    best_set = np.empty((0, 3), dtype=np.int64)
    best_active_spheres = []

    np.random.seed(42)

    # Phase 1: Greedy Restarts
    num_restarts = 50
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
        order = np.random.permutation(num_points_total)
        curr_set, curr_spheres = greedy_build(order)
        if len(curr_set) > len(best_set):
            best_set = curr_set.copy()
            best_active_spheres = curr_spheres

    # Phase 2: Local Search with Bitset/Sphere Maintenance
    if len(best_set) == 0:
        return best_set

    current_indices = set(best_set.tolist())
    active_spheres = best_active_spheres.copy()
    
    # Map index to position in active_spheres? No, we scan.
    # To speed up removal, we can store a mapping: point_idx -> list of sphere_indices in active_spheres containing it.
    point_to_spheres = {idx: [] for idx in current_indices}
    for s_idx, (params, indices) in enumerate(active_spheres):
        for p_idx in indices:
            point_to_spheres[p_idx].append(s_idx)

    def remove_point_from_spheres(p_idx):
        """Remove all spheres containing p_idx from active_spheres."""
        spheres_to_remove = set(point_to_spheres.get(p_idx, []))
        if not spheres_to_remove:
            return
            
        # Update point_to_spheres for other points in these spheres
        new_active_spheres = []
        for s_idx, (params, indices) in enumerate(active_spheres):
            if s_idx in spheres_to_remove:
                # This sphere is removed
                for other_idx in indices:
                    if other_idx != p_idx:
                        # Remove s_idx from other_idx's list
                        # This is slow if lists are long, but m is small
                        if s_idx in point_to_spheres[other_idx]:
                            point_to_spheres[other_idx].remove(s_idx)
            else:
                # Keep sphere, but re-index? 
                # Re-indexing is O(N). 
                # Since we are modifying active_spheres, indices will change.
                # This is a problem.
                # Instead of re-indexing, we can just use a "valid" flag or rebuild the mapping.
                pass
        
        # Simpler approach: Rebuild point_to_spheres after every removal.
        # This is O(m^4) which is fine for m < 30.
        pass

    # Given the complexity of maintaining the mapping, and the fact that m is small,
    # we might just recompute the spheres list from scratch when removing a point?
    # No, that's O(m^4) compute_sphere calls.
    
    # Let's stick to the simpler "Recalculate Determinants" approach but optimize the Python overhead.
    # The previous program was 1.1111.
    # The main bottleneck was likely `itertools.combinations` inside the loop.
    
    # Let's try to improve the Local Search by increasing the number of swaps and using a better heuristic.
    
    current_set = best_set.copy()
    current_indices_set = set(current_set.tolist())
    remaining_indices = list(set(range(num_points_total)) - current_indices_set)
    
    def is_valid_addition(arr: np.ndarray, p: np.ndarray) -> bool:
        m = len(arr)
        if m < 4:
            return True
        diff = arr.astype(np.float64) - p.astype(np.float64)
        sq_diff = (arr.astype(np.float64) ** 2).sum(axis=1) - (p.astype(np.float64) ** 2).sum()
        M = np.empty((m, 4), dtype=np.float64)
        M[:, 0:3] = diff
        M[:, 3] = sq_diff
        combo_indices = np.array(list(itertools.combinations(range(m), 4)), dtype=np.int32)
        if len(combo_indices) == 0:
            return True
        batch_size = 10000
        for start in range(0, len(combo_indices), batch_size):
            end = min(start + batch_size, len(combo_indices))
            M_sub = M[combo_indices[start:end]]
            dets = np.linalg.det(M_sub)
            if np.any(np.abs(dets) < 0.5):
                return False
        return True

    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try to Add
        if remaining_indices:
            # Shuffle to randomize
            np.random.shuffle(remaining_indices)
            for idx in remaining_indices:
                p = grid[idx]
                if is_valid_addition(current_set, p):
                    current_set = np.vstack([current_set, p])
                    current_indices_set.add(idx)
                    remaining_indices.remove(idx)
                    improved = True
                    no_improve_count = 0
                    break
        
        # 2. Try to Swap
        if not improved and len(current_indices_set) > 0 and remaining_indices:
            swaps_to_try = 50
            for _ in range(swaps_to_try):
                if time.time() > deadline:
                    break
                in_idx_val = list(current_indices_set)[np.random.randint(len(current_indices_set))]
                out_idx_val = remaining_indices[np.random.randint(len(remaining_indices))]
                
                in_pos = -1
                for i in range(len(current_set)):
                    if current_set[i][0] == grid[in_idx_val][0] and current_set[i][1] == grid[in_idx_val][1] and current_set[i][2] == grid[in_idx_val][2]:
                        in_pos = i
                        break
                if in_pos == -1:
                    continue
                    
                reduced_mask = np.ones(len(current_set), dtype=bool)
                reduced_mask[in_pos] = False
                reduced_set = current_set[reduced_mask]
                p_out = grid[out_idx_val]
                
                if is_valid_addition(reduced_set, p_out):
                    current_set = np.vstack([reduced_set, p_out])
                    current_indices_set.remove(in_idx_val)
                    current_indices_set.add(out_idx_val)
                    remaining_indices.remove(out_idx_val)
                    remaining_indices.append(in_idx_val)
                    improved = True
                    no_improve_count = 0
                    break

        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                if len(current_indices_set) > 10:
                    kick_size = min(5, max(3, len(current_indices_set) // 6))
                    kick_idxs = np.random.choice(list(current_indices_set), kick_size, replace=False)
                    for idx in kick_idxs:
                        pos = -1
                        for i in range(len(current_set)):
                            if current_set[i][0] == grid[idx][0] and current_set[i][1] == grid[idx][1] and current_set[i][2] == grid[idx][2]:
                                pos = i
                                break
                        if pos != -1:
                            mask = np.ones(len(current_set), dtype=bool)
                            mask[pos] = False
                            current_set = current_set[mask]
                        current_indices_set.remove(idx)
                        remaining_indices.append(idx)
                    no_improve_count = 0
                    max_no_improve = 200

    return current_set.astype(np.int64)
