<<<<<<< SEARCH
def _spherical_lloyd_iteration(points, max_iter=150, time_limit=30, damping=0.4):
    """
    Spherical Lloyd's algorithm (Voronoi relaxation) - vectorized version.
    For each point, compute the average of outward normals of adjacent facets,
    then move the point toward that direction.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    current_cost = _covering_radius(points)
    best_points = points.copy()
    best_cost = current_cost
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        try:
            hull = ConvexHull(points)
        except:
            break
        
        simplices = hull.simplices
        p1 = points[simplices[:, 0]]
        p2 = points[simplices[:, 1]]
        p3 = points[simplices[:, 2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n, axis=1, keepdims=True)
        mask = norm_n[:, 0] > 1e-10
        if not np.any(mask):
            break
        
        n_valid = n[mask] / norm_n[mask]
        p1_valid = p1[mask]
        simplices_valid = simplices[mask]
        
        # Ensure normals point outward
        dots = np.sum(n_valid * p1_valid, axis=1)
        n_valid[dots < 0] *= -1
        
        # Accumulate normal contributions for each point
        normal_sum = np.zeros((N, 3))
        
        # For each facet, add its normal to all three vertices
        for fi in range(len(simplices_valid)):
            fi_idx = fi  # index into valid arrays
            s = simplices_valid[fi_idx]
            nv = n_valid[fi_idx]
            normal_sum[s[0]] += nv
            normal_sum[s[1]] += nv
            normal_sum[s[2]] += nv
        
        # Compute new positions
        norms_ns = np.linalg.norm(normal_sum, axis=1, keepdims=True)
        valid_mask = norms_ns[:, 0] > 1e-10
        
        new_points = points.copy()
        for pi in range(N):
            if valid_mask[pi]:
                target = normal_sum[pi] / norms_ns[pi, 0]
                new_points[pi] = points[pi] + damping * (target - points[pi])
                new_points[pi] = new_points[pi] / np.linalg.norm(new_points[pi])
        
        new_points = _normalize(new_points)
        new_cost = _covering_radius(new_points)
        
        if new_cost < current_cost:
            current_cost = new_cost
            points = new_points
            if current_cost < best_cost:
                best_cost = current_cost
                best_points = points.copy()
        else:
            # Try smaller damping
            if i > 20:
                damping *= 0.9
            else:
                break
    
    return best_points
=======
def _spherical_lloyd_iteration(points, max_iter=150, time_limit=30, damping=0.4):
    """
    Spherical Lloyd's algorithm (Voronoi relaxation) - fully vectorized version.
    For each point, compute the average of outward normals of adjacent facets,
    then move the point toward that direction.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    current_cost = _covering_radius(points)
    best_points = points.copy()
    best_cost = current_cost
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        try:
            hull = ConvexHull(points)
        except:
            break
        
        simplices = hull.simplices
        p1 = points[simplices[:, 0]]
        p2 = points[simplices[:, 1]]
        p3 = points[simplices[:, 2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n, axis=1, keepdims=True)
        mask = norm_n[:, 0] > 1e-10
        if not np.any(mask):
            break
        
        n_valid = n[mask] / norm_n[mask]
        p1_valid = p1[mask]
        simplices_valid = simplices[mask]
        
        # Ensure normals point outward
        dots = np.sum(n_valid * p1_valid, axis=1)
        n_valid[dots < 0] *= -1
        
        # Vectorized accumulation: add each facet normal to its 3 vertices
        normal_sum = np.zeros((N, 3))
        np.add.at(normal_sum, simplices_valid[:, 0], n_valid)
        np.add.at(normal_sum, simplices_valid[:, 1], n_valid)
        np.add.at(normal_sum, simplices_valid[:, 2], n_valid)
        
        # Compute new positions
        norms_ns = np.linalg.norm(normal_sum, axis=1, keepdims=True)
        valid_mask = norms_ns[:, 0] > 1e-10
        
        new_points = points.copy()
        target = np.where(valid_mask, normal_sum / np.maximum(norms_ns, 1e-10), points)
        new_points = points + damping * (target - points)
        new_points = _normalize(new_points)
        
        new_cost = _covering_radius(new_points)
        
        if new_cost < current_cost:
            current_cost = new_cost
            points = new_points
            if current_cost < best_cost:
                best_cost = current_cost
                best_points = points.copy()
        else:
            # Try smaller damping
            if i > 20:
                damping *= 0.9
            else:
                break
    
    return best_points
>>>>>>> REPLACE

