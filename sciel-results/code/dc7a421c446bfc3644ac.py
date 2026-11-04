def polish(points, time_budget, start_alpha=0.05):
    """
    High-precision polish pass.
    Phase 1: Gradient descent on smooth surrogate.
    Phase 2: Targeted facet push.
    """
    start = time.time()
    alpha = start_alpha
    best_local = points.copy()
    best_local_radius = np.pi

    # Phase 1: Gradient Descent
    for step in range(1000):
        if time.time() - start > time_budget * 0.7:
            break
            
        radii, centers, simplices = compute_facet_data(points)
        if len(radii) == 0:
            break

        val, grad = smooth_objective_and_grad(points, radii, centers, simplices)
        
        if len(radii) > 0:
            max_r = np.max(radii)
            if max_r < best_local_radius:
                best_local_radius = max_r
                best_local = points.copy()
                
        # Project gradient to tangent space
        grad_tangent = grad - np.sum(grad * points, axis=1, keepdims=True) * points
        
        # Simple line search to ensure descent
        current_val = val
        step_size = alpha
        improved = False
        
        for ls_iter in range(3): # Reduce line search iterations
            new_points = points + step_size * grad_tangent
            new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
            
            # Quick check: just compute max radius to save time? 
            # Or compute smooth objective again?
            # Let's compute max radius for speed, but accept it might not be smooth descent.
            # Actually, to ensure descent in the surrogate, we should evaluate the surrogate.
            # But that requires another hull. 
            # Compromise: evaluate max radius. If it decreases, accept.
            new_radii, _, _ = compute_facet_data(new_points)
            if len(new_radii) == 0:
                break
            new_max_r = np.max(new_radii)
            
            if new_max_r < max_r: # Use max radius as proxy for objective in line search
                points = new_points
                improved = True
                break
            else:
                step_size *= 0.5
        
        if not improved:
            if step_size < 1e-8:
                break
            points = points + step_size * grad_tangent
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            
        alpha *= 0.99
        if alpha < 1e-6:
            break
            
    points = best_local
    
    # Phase 2: Facet Push with decaying step
    alpha_push = 0.02
    for step in range(500):
        if time.time() - start > time_budget:
            break
            
        radii, centers, simplices = compute_facet_data(points)
        if len(radii) == 0:
            break
            
        max_r = np.max(radii)
        if max_r < best_local_radius:
            best_local_radius = max_r
            best_local = points.copy()
            
        # Target top 10% largest facets
        sorted_indices = np.argsort(radii)[::-1]
        num_top = max(1, len(radii) // 10)
        top_mask = np.zeros(len(radii), dtype=bool)
        top_mask[sorted_indices[:num_top]] = True
        
        updates = np.zeros_like(points)
        
        # Vectorized update for top facets
        top_idx = np.where(top_mask)[0]
        if len(top_idx) > 0:
            top_simplices = simplices[top_idx]
            top_centers = centers[top_idx]
            top_radii = radii[top_idx]
            
            # Step size per facet
            steps = alpha_push * top_radii[:, np.newaxis] # (K, 1)
            # Direction: center - point
            # For each facet m, vertices v, update += step[m] * (center[m] - point[v])
            
            # This is hard to vectorize perfectly without loops because points[v] varies.
            # But we can use np.add.at
            # values[m, k] = step[m] * (center[m] - points[simplices[m, k]])
            
            # Let's just loop over top facets, it's only ~20% of them
            for m in top_idx:
                idx = simplices[m]
                i_idx, j_idx, k_idx = idx
                center = centers[m]
                r = radii[m]
                
                step_size = alpha_push * r
                
                updates[i_idx] += step_size * (center - points[i_idx])
                updates[j_idx] += step_size * (center - points[j_idx])
                updates[k_idx] += step_size * (center - points[k_idx])
            
        points = points + updates
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        alpha_push *= 0.99
        if alpha_push < 1e-6:
            break
            
    return best_local
