<<<<<<< SEARCH
def _simulated_annealing(bary_pts, rng, start_temp=0.05, end_temp=0.001, steps=5000):
    """Perform Simulated Annealing on barycentric coordinates."""
    current_pts = bary_pts.copy()
    current_areas = _compute_areas(current_pts)
    current_min_area = np.min(current_areas)
    current_score = -current_min_area  # We want to maximize min area, so minimize -min area
    
    best_pts = current_pts.copy()
    best_score = current_score
    
    # Precompute neighbor indices? No, random perturbations are better for global search.
    
    for i in range(steps):
        # Cooling schedule
        temp = start_temp * (end_temp / start_temp) ** (i / steps)
        
        # Generate a neighbor: perturb one or a few points
        n_perturb = rng.integers(1, 4)  # Perturb 1 to 3 points
        idx = rng.choice(N, n_perturb, replace=False)
        
        new_pts = current_pts.copy()
        # Gaussian perturbation
        scale = 0.1 * temp + 0.01
        new_pts[idx] += rng.normal(0, scale, size=(n_perturb, 2))
        new_pts = _project_bary_to_simplex(new_pts)
        
        new_areas = _compute_areas(new_pts)
        new_min_area = np.min(new_areas)
        new_score = -new_min_area
        
        delta = new_score - current_score
        
        # Acceptance probability
        if delta < 0:
            if rng.random() < np.exp(-delta / temp):
                current_pts = new_pts
                current_min_area = new_min_area
                current_score = new_score
        else:
            current_pts = new_pts
            current_min_area = new_min_area
            current_score = new_score
        
        if current_score < best_score:
            best_score = current_score
            best_pts = current_pts.copy()
            
    return best_pts, -best_score
=======
def _gradient_descent_repulsion(bary_pts, rng, steps=2000, lr_init=0.05, lr_decay=0.999):
    """
    Perform gradient descent with explicit repulsion for small triangles.
    This implements a 'Kinetic Annealing with Topological Collision Resolution' style update.
    """
    current_pts = bary_pts.copy()
    current_min_area = np.min(_compute_areas(current_pts))
    best_pts = current_pts.copy()
    best_area = current_min_area
    
    lr = lr_init
    
    for i in range(steps):
        areas = _compute_areas(current_pts)
        min_idx = np.argmin(areas)
        min_area = areas[min_idx]
        
        # If min area is very small, we need strong repulsion
        # Calculate gradient of min area (approximate)
        # Instead of full gradient, we use a heuristic:
        # Push the 3 points of the min triangle apart along their normals or just random directions away from centroid
        # Better: Compute gradient of the area with respect to points
        
        # Let's use a simple gradient step for the min triangle area
        # Area = 0.5 * | (Pj - Pi) x (Pk - Pi) |
        # For barycentric, it's linear in coordinates.
        
        # Get indices of the min triangle
        i_idx, j_idx, k_idx = _TRIPLETS[min_idx]
        
        pi = current_pts[i_idx]
        pj = current_pts[j_idx]
        pk = current_pts[k_idx]
        
        # Compute gradients for these 3 points
        # dA/dPi, dA/dPj, dA/dPk
        # Using the determinant formula from _compute_areas
        # This is complex to derive manually for all points, so we use a finite difference or heuristic push.
        
        # Heuristic Push:
        # Move i, j, k away from the centroid of the triangle
        centroid = (pi + pj + pk) / 3.0
        
        # Direction away from centroid
        dir_i = pi - centroid
        dir_j = pj - centroid
        dir_k = pk - centroid
        
        # Normalize
        norm_i = np.linalg.norm(dir_i)
        norm_j = np.linalg.norm(dir_j)
        norm_k = np.linalg.norm(dir_k)
        
        if norm_i > 1e-9: dir_i /= norm_i
        else: dir_i = np.array([0.1, 0.1])
        if norm_j > 1e-9: dir_j /= norm_j
        else: dir_j = np.array([0.1, 0.1])
        if norm_k > 1e-9: dir_k /= norm_k
        else: dir_k = np.array([0.1, 0.1])
        
        # Also add global repulsion: push all points apart slightly to avoid clustering
        # Simple O(N^2) repulsion is too slow for 2000 steps * 11 points? 11*11=121, so it's fine.
        grad = np.zeros_like(current_pts)
        
        # Repulsion term
        for a in range(N):
            for b in range(a + 1, N):
                diff = current_pts[a] - current_pts[b]
                dist_sq = np.dot(diff, diff)
                if dist_sq < 1e-6:
                    # Random kick if too close
                    grad[a] += rng.normal(0, 0.1, 2)
                    grad[b] -= rng.normal(0, 0.1, 2)
                else:
                    # Repulsive force ~ 1/r^2
                    force = 1.0 / (dist_sq + 1e-4)
                    grad[a] += force * diff
                    grad[b] -= force * diff
        
        # Specific push for min triangle
        # We want to increase the area, so we move points in the direction that increases area.
        # Moving away from centroid generally increases area if points are on the perimeter of the convex hull of the 3 points?
        # Actually, for a triangle, moving vertices radially out from centroid increases area.
        push_scale = lr * 0.5
        grad[i_idx] += dir_i * push_scale
        grad[j_idx] += dir_j * push_scale
        grad[k_idx] += dir_k * push_scale
        
        # Add random noise for exploration
        grad += rng.normal(0, lr * 0.1, size=grad.shape)
        
        # Update
        new_pts = current_pts + grad
        new_pts = _project_bary_to_simplex(new_pts)
        
        new_min_area = np.min(_compute_areas(new_pts))
        
        if new_min_area > current_min_area - 1e-6: # Accept if not worse (with tolerance)
            current_pts = new_pts
            current_min_area = new_min_area
            
            if current_min_area > best_area:
                best_area = current_min_area
                best_pts = current_pts.copy()
        
        lr *= lr_decay
        
    return best_pts, best_area
>>>>>>> REPLACE

