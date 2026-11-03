<<<<<<< SEARCH
def _refine_covering(points, n_iterations=3000):
    """Refine points to minimize covering radius directly."""
    points = points.copy()
    n = len(points)
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    for it in range(n_iterations):
        # Compute pairwise distances
        dots = points @ points.T
        np.fill_diagonal(dots, 0.0)
        dots = np.clip(dots, -0.9999, 0.9999)
        
        # Find nearest neighbors for each point
        min_dots = np.min(dots, axis=1)  # Most negative dot = farthest
        # Actually we want the closest: max dot
        max_dots = np.max(dots, axis=1)
        
        # Force: move away from nearest neighbors
        # For each point, find its 3 nearest neighbors and move away from them
        force = np.zeros_like(points)
        
        for i in range(n):
            # Find 3 nearest neighbors
            dists = 1.0 - dots[i]
            idx = np.argsort(dists)[:3]
            
            for j in idx:
                if i == j:
                    continue
                # Move away from point j
                perp = points[i] - dots[i, j] * points[j]
                perp_norm = np.linalg.norm(perp)
                if perp_norm > 1e-15:
                    force[i] += perp / perp_norm
        
        # Normalize force
        force_norms = np.linalg.norm(force, axis=1, keepdims=True)
        force_norms = np.maximum(force_norms, 1e-15)
        force_unit = force / force_norms
        
        # Learning rate
        lr = 0.02 * (1 - it / n_iterations)
        
        # Update
        points = points + lr * force_unit
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        if it % 100 == 0:
            radius = _covering_radius(points)
            if radius < best_radius:
                best_radius = radius
                best_points = points.copy()
    
    return best_points, best_radius
=======
def _refine_covering(points, n_iterations=3000):
    """Refine points to minimize covering radius directly using KDTree."""
    points = points.copy()
    n = len(points)
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    for it in range(n_iterations):
        # Build KDTree for efficient nearest neighbor search
        tree = cKDTree(points)
        
        # Find 3 nearest neighbors for each point
        # k=4 to include self, then skip self
        dists, idxs = tree.query(points, k=4)
        
        force = np.zeros_like(points)
        
        for i in range(n):
            # idxs[i] is array of 4 indices: self, n1, n2, n3
            # Skip self (index 0)
            for j_idx in range(1, 4):
                j = idxs[i, j_idx]
                # Move away from point j
                # Vector from j to i
                v = points[i] - points[j]
                v_norm = np.linalg.norm(v)
                if v_norm > 1e-10:
                    # Project v onto tangent plane at i
                    # v_tan = v - (v.p_i) p_i
                    dot_vp = np.dot(v, points[i])
                    v_tan = v - dot_vp * points[i]
                    v_tan_norm = np.linalg.norm(v_tan)
                    if v_tan_norm > 1e-10:
                        force[i] += v_tan / v_tan_norm
        
        # Normalize force
        force_norms = np.linalg.norm(force, axis=1, keepdims=True)
        force_norms = np.maximum(force_norms, 1e-15)
        force_unit = force / force_norms
        
        # Learning rate with decay
        lr = 0.05 * (1 - it / n_iterations) ** 2
        
        # Update
        points = points + lr * force_unit
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        if it % 50 == 0:
            radius = _covering_radius(points)
            if radius < best_radius:
                best_radius = radius
                best_points = points.copy()
    
    return best_points, best_radius
>>>>>>> REPLACE

