<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = 120.0  # seconds
    
    # Get initial hex lattice configuration
    points, init_r = _hex_lattice_init(N)
    
    best_pts = points.copy()
    best_r = init_r
    
    # Strategy: Use L-BFGS-B to maximize the minimum distance directly
    # Objective: minimize -min_pairwise_distance (or equivalently, maximize it)
    # But we need to also respect the boundary constraint
    
    # Better: minimize a smooth objective that penalizes both overlap and boundary violation
    # Using soft-min for the minimum distance
    
    def soft_min_obj(x, alpha=50.0):
        """Smooth approximation of -min_pairwise_dist + boundary_penalty."""
        pts = x.reshape(-1, 2)
        n = len(pts)
        
        # Compute all pairwise distances
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        
        # Soft-min: -log(sum(exp(-alpha * d))) / alpha approximates min(d)
        # To avoid overflow, shift by min value
        d_flat = d[d < np.inf]
        if len(d_flat) == 0:
            return 0.0
        
        d_min = np.min(d_flat)
        # Soft min approximation
        log_sum = np.log(np.sum(np.exp(-alpha * (d_flat - d_min))))
        soft_min = d_min - log_sum / alpha
        
        # Boundary penalty
        norms = np.hypot(pts[:, 0], pts[:, 1])
        boundary_violation = np.maximum(0.0, norms - (1.0 - soft_min / 2.0))
        boundary_pen = np.sum(boundary_violation ** 2)
        
        return -soft_min + 10.0 * boundary_pen
    
    def soft_min_grad(x, alpha=50.0):
        """Gradient of soft_min_obj."""
        pts = x.reshape(-1, 2)
        n = len(pts)
        
        # Compute all pairwise distances
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        
        d_flat = d[d < np.inf]
        if len(d_flat) == 0:
            return np.zeros_like(x)
        
        d_min = np.min(d_flat)
        
        # Soft min: f = d_min - log(sum(exp(-alpha*(d - d_min)))) / alpha
        # grad f = -1/alpha * sum(exp(-alpha*(d-d_min)) * (-alpha * grad d)) / sum(exp(-alpha*(d-d_min)))
        #         = sum(exp(-alpha*(d-d_min)) * grad d) / sum(exp(-alpha*(d-d_min)))
        
        # We need grad of each d_ij with respect to pts
        # d_ij = ||p_i - p_j||
        # grad_{p_i} d_ij = (p_i - p_j) / d_ij
        # grad_{p_j} d_ij = (p_j - p_i) / d_ij
        
        # For efficiency, compute weights
        exp_terms = np.exp(-alpha * (d - d_min))
        np.fill_diagonal(exp_terms, 0.0)
        w_sum = np.sum(exp_terms)
        
        if w_sum == 0:
            return np.zeros_like(x)
        
        # Gradient of soft_min w.r.t. points
        # grad_{p_i} soft_min = sum_j w_ij * (p_i - p_j) / d_ij / w_sum
        # where w_ij = exp(-alpha * (d_ij - d_min))
        
        grad_pts = np.zeros_like(pts)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                w_ij = exp_terms[i, j]
                d_ij = d[i, j]
                if d_ij > 1e-10:
                    diff = pts[i] - pts[j]
                    grad_pts[i] += w_ij * diff / d_ij
                    grad_pts[j] -= w_ij * diff / d_ij
        
        grad_pts /= w_sum
        grad_soft_min = -grad_pts  # because obj = -soft_min
        
        # Boundary penalty gradient
        norms = np.hypot(pts[:, 0], pts[:, 1])
        boundary_violation = np.maximum(0.0, norms - (1.0 - soft_min / 2.0))
        boundary_pen = np.sum(boundary_violation ** 2)
        
        # Grad of boundary penalty
        grad_boundary = np.zeros_like(pts)
        for i in range(n):
            if boundary_violation[i] > 0:
                norm_i = norms[i]
                if norm_i > 1e-10:
                    grad_boundary[i] = 2.0 * boundary_violation[i] * pts[i] / norm_i
        
        grad_obj = grad_soft_min + 10.0 * grad_boundary
        
        return grad_obj.flatten()
    
    # Optimize using L-BFGS-B
    x_init = points.flatten()
    
    # Bounds: keep points within unit circle
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    remaining_time = deadline - (time.time() - start_time)
    
    if remaining_time > 20:
        try:
            result = minimize(
                soft_min_obj,
                x_init,
                jac=soft_min_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 1000}
            )
            
            if result.success or result.fun < soft_min_obj(x_init):
                candidate_pts = result.x.reshape(-1, 2)
                candidate_r = _compute_score(candidate_pts)
                
                if candidate_r > best_r:
                    best_pts = candidate_pts
                    best_r = candidate_r
        except Exception:
            pass
    
    # Second pass with different alpha for finer tuning
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time > 15:
        def soft_min_obj_fine(x, alpha=100.0):
            pts = x.reshape(-1, 2)
            n = len(pts)
            
            d = cdist(pts, pts)
            np.fill_diagonal(d, np.inf)
            
            d_flat = d[d < np.inf]
            if len(d_flat) == 0:
                return 0.0
            
            d_min = np.min(d_flat)
            log_sum = np.log(np.sum(np.exp(-alpha * (d_flat - d_min))))
            soft_min = d_min - log_sum / alpha
            
            norms = np.hypot(pts[:, 0], pts[:, 1])
            boundary_violation = np.maximum(0.0, norms - (1.0 - soft_min / 2.0))
            boundary_pen = np.sum(boundary_violation ** 2)
            
            return -soft_min + 10.0 * boundary_pen
        
        def soft_min_grad_fine(x, alpha=100.0):
            pts = x.reshape(-1, 2)
            n = len(pts)
            
            d = cdist(pts, pts)
            np.fill_diagonal(d, np.inf)
            
            d_flat = d[d < np.inf]
            if len(d_flat) == 0:
                return np.zeros_like(x)
            
            d_min = np.min(d_flat)
            
            exp_terms = np.exp(-alpha * (d - d_min))
            np.fill_diagonal(exp_terms, 0.0)
            w_sum = np.sum(exp_terms)
            
            if w_sum == 0:
                return np.zeros_like(x)
            
            grad_pts = np.zeros_like(pts)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    w_ij = exp_terms[i, j]
                    d_ij = d[i, j]
                    if d_ij > 1e-10:
                        diff = pts[i] - pts[j]
                        grad_pts[i] += w_ij * diff / d_ij
                        grad_pts[j] -= w_ij * diff / d_ij
            
            grad_pts /= w_sum
            grad_soft_min = -grad_pts
            
            norms = np.hypot(pts[:, 0], pts[:, 1])
            soft_min = d_min - np.log(np.sum(np.exp(-alpha * (d_flat - d_min)))) / alpha
            boundary_violation = np.maximum(0.0, norms - (1.0 - soft_min / 2.0))
            
            grad_boundary = np.zeros_like(pts)
            for i in range(n):
                if boundary_violation[i] > 0:
                    norm_i = norms[i]
                    if norm_i > 1e-10:
                        grad_boundary[i] = 2.0 * boundary_violation[i] * pts[i] / norm_i
            
            grad_obj = grad_soft_min + 10.0 * grad_boundary
            return grad_obj.flatten()
        
        x_init2 = best_pts.flatten()
        try:
            result2 = minimize(
                soft_min_obj_fine,
                x_init2,
                jac=soft_min_grad_fine,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 600}
            )
            
            if result2.success or result2.fun < soft_min_obj_fine(x_init2):
                candidate_pts = result2.x.reshape(-1, 2)
                candidate_r = _compute_score(candidate_pts)
                
                if candidate_r > best_r:
                    best_pts = candidate_pts
                    best_r = candidate_r
        except Exception:
            pass
    
    # Final check: ensure all points are within bounds and valid
    final_pts = best_pts.copy()
    
    # Clamp points to be within unit circle (with small margin for radius)
    norms = np.hypot(final_pts[:, 0], final_pts[:, 1])
    max_allowed_norm = 1.0 - best_r
    for i in range(len(final_pts)):
        if norms[i] > max_allowed_norm:
            if norms[i] > 1e-10:
                final_pts[i] = final_pts[i] * (max_allowed_norm / norms[i])
            else:
                final_pts[i] = [0.0, 0.0]
    
    return final_pts
=======
def construct():
    start_time = time.time()
    
    # Get initial hex lattice configuration
    points, init_r = _hex_lattice_init(N)
    
    best_pts = points.copy()
    best_r = init_r
    
    # Phase 1: Local refinement via coordinate descent
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 10:
        refined_pts, refined_r = _refine_local(best_pts, best_r, time_budget=min(20.0, remaining_time - 5))
        if refined_r > best_r:
            best_pts = refined_pts
            best_r = refined_r
    
    # Phase 2: L-BFGS-B with soft-min objective for global adjustment
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 25:
        def soft_min_obj(x, alpha=50.0):
            """Smooth approximation of -min_pairwise_dist + boundary_penalty."""
            pts = x.reshape(-1, 2)
            n = len(pts)
            
            # Compute all pairwise distances
            d = cdist(pts, pts)
            np.fill_diagonal(d, np.inf)
            
            d_flat = d[d < np.inf]
            if len(d_flat) == 0:
                return 0.0
            
            d_min = np.min(d_flat)
            log_sum = np.log(np.sum(np.exp(-alpha * (d_flat - d_min))))
            soft_min = d_min - log_sum / alpha
            
            # Boundary penalty
            norms = np.hypot(pts[:, 0], pts[:, 1])
            boundary_violation = np.maximum(0.0, norms - (1.0 - soft_min / 2.0))
            boundary_pen = np.sum(boundary_violation ** 2)
            
            return -soft_min + 10.0 * boundary_pen
        
        def soft_min_grad(x, alpha=50.0):
            """Gradient of soft_min_obj."""
            pts = x.reshape(-1, 2)
            n = len(pts)
            
            d = cdist(pts, pts)
            np.fill_diagonal(d, np.inf)
            
            d_flat = d[d < np.inf]
            if len(d_flat) == 0:
                return np.zeros_like(x)
            
            d_min = np.min(d_flat)
            
            exp_terms = np.exp(-alpha * (d - d_min))
            np.fill_diagonal(exp_terms, 0.0)
            w_sum = np.sum(exp_terms)
            
            if w_sum == 0:
                return np.zeros_like(x)
            
            grad_pts = np.zeros_like(pts)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    w_ij = exp_terms[i, j]
                    d_ij = d[i, j]
                    if d_ij > 1e-10:
                        diff = pts[i] - pts[j]
                        grad_pts[i] += w_ij * diff / d_ij
                        grad_pts[j] -= w_ij * diff / d_ij
            
            grad_pts /= w_sum
            grad_soft_min = -grad_pts
            
            norms = np.hypot(pts[:, 0], pts[:, 1])
            soft_min_val = d_min - np.log(np.sum(np.exp(-alpha * (d_flat - d_min)))) / alpha
            boundary_violation = np.maximum(0.0, norms - (1.0 - soft_min_val / 2.0))
            
            grad_boundary = np.zeros_like(pts)
            for i in range(n):
                if boundary_violation[i] > 0:
                    norm_i = norms[i]
                    if norm_i > 1e-10:
                        grad_boundary[i] = 2.0 * boundary_violation[i] * pts[i] / norm_i
            
            grad_obj = grad_soft_min + 10.0 * grad_boundary
            return grad_obj.flatten()
        
        x_init = best_pts.flatten()
        bounds = [(-1.0, 1.0)] * (2 * N)
        
        try:
            result = minimize(
                soft_min_obj,
                x_init,
                jac=soft_min_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 500}
            )
            
            if result.success or result.fun < soft_min_obj(x_init):
                candidate_pts = result.x.reshape(-1, 2)
                candidate_r = _compute_score(candidate_pts)
                
                if candidate_r > best_r:
                    best_pts = candidate_pts
                    best_r = candidate_r
        except Exception:
            pass
    
    # Phase 3: Fine-tuning with higher alpha
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 15:
        def soft_min_obj_fine(x, alpha=100.0):
            pts = x.reshape(-1, 2)
            n = len(pts)
            
            d = cdist(pts, pts)
            np.fill_diagonal(d, np.inf)
            
            d_flat = d[d < np.inf]
            if len(d_flat) == 0:
                return 0.0
            
            d_min = np.min(d_flat)
            log_sum = np.log(np.sum(np.exp(-alpha * (d_flat - d_min))))
            soft_min = d_min - log_sum / alpha
            
            norms = np.hypot(pts[:, 0], pts[:, 1])
            boundary_violation = np.maximum(0.0, norms - (1.0 - soft_min / 2.0))
            boundary_pen = np.sum(boundary_violation ** 2)
            
            return -soft_min + 10.0 * boundary_pen
        
        def soft_min_grad_fine(x, alpha=100.0):
            pts = x.reshape(-1, 2)
            n = len(pts)
            
            d = cdist(pts, pts)
            np.fill_diagonal(d, np.inf)
            
            d_flat = d[d < np.inf]
            if len(d_flat) == 0:
                return np.zeros_like(x)
            
            d_min = np.min(d_flat)
            
            exp_terms = np.exp(-alpha * (d - d_min))
            np.fill_diagonal(exp_terms, 0.0)
            w_sum = np.sum(exp_terms)
            
            if w_sum == 0:
                return np.zeros_like(x)
            
            grad_pts = np.zeros_like(pts)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    w_ij = exp_terms[i, j]
                    d_ij = d[i, j]
                    if d_ij > 1e-10:
                        diff = pts[i] - pts[j]
                        grad_pts[i] += w_ij * diff / d_ij
                        grad_pts[j] -= w_ij * diff / d_ij
            
            grad_pts /= w_sum
            grad_soft_min = -grad_pts
            
            norms = np.hypot(pts[:, 0], pts[:, 1])
            soft_min_val = d_min - np.log(np.sum(np.exp(-alpha * (d_flat - d_min)))) / alpha
            boundary_violation = np.maximum(0.0, norms - (1.0 - soft_min_val / 2.0))
            
            grad_boundary = np.zeros_like(pts)
            for i in range(n):
                if boundary_violation[i] > 0:
                    norm_i = norms[i]
                    if norm_i > 1e-10:
                        grad_boundary[i] = 2.0 * boundary_violation[i] * pts[i] / norm_i
            
            grad_obj = grad_soft_min + 10.0 * grad_boundary
            return grad_obj.flatten()
        
        x_init2 = best_pts.flatten()
        try:
            result2 = minimize(
                soft_min_obj_fine,
                x_init2,
                jac=soft_min_grad_fine,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 400}
            )
            
            if result2.success or result2.fun < soft_min_obj_fine(x_init2):
                candidate_pts = result2.x.reshape(-1, 2)
                candidate_r = _compute_score(candidate_pts)
                
                if candidate_r > best_r:
                    best_pts = candidate_pts
                    best_r = candidate_r
        except Exception:
            pass
    
    # Phase 4: Final local polish
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 5:
        final_pts, final_r = _refine_local(best_pts, best_r, time_budget=min(5.0, remaining_time - 2))
        if final_r > best_r:
            best_pts = final_pts
            best_r = final_r
    
    # Final check: ensure all points are within bounds and valid
    final_pts = best_pts.copy()
    
    # Clamp points to be within unit circle (with small margin for radius)
    norms = np.hypot(final_pts[:, 0], final_pts[:, 1])
    max_allowed_norm = 1.0 - best_r
    for i in range(len(final_pts)):
        if norms[i] > max_allowed_norm:
            if norms[i] > 1e-10:
                final_pts[i] = final_pts[i] * (max_allowed_norm / norms[i])
            else:
                final_pts[i] = [0.0, 0.0]
    
    return final_pts
>>>>>>> REPLACE

