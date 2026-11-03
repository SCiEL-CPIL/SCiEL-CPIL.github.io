<<<<<<< SEARCH
def _optimize_facet_shrink(pts, max_iter=1000, initial_step=0.5, decay=0.995, beta_start=2.0, beta_end=20.0):
    """
    Optimizes the configuration by moving each point towards the weighted
    average of the circumcenters of its incident facets.
    Uses adaptive step size and increasing beta to focus on worst facets.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    step_size = initial_step
    
    for it in range(max_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        weights_sum = np.zeros(n)
        vec_sum = np.zeros((n, 3))
        
        max_cr_global = 0.0
        facet_data = []

        for simplex in hull.simplices:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            v = np.cross(a - b, b - c)
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-12:
                continue
            u = v / norm_v
            
            n_face = np.cross(b - a, c - a)
            norm_n = np.linalg.norm(n_face)
            if norm_n < 1e-12:
                continue
            n_face = n_face / norm_n
            if np.dot(n_face, a) < 0:
                n_face = -n_face
                
            if np.dot(u, n_face) > 0:
                cc = u
            else:
                cc = -u
                
            d_a = np.arccos(np.clip(np.dot(cc, a), -1, 1))
            d_b = np.arccos(np.clip(np.dot(cc, b), -1, 1))
            d_c = np.arccos(np.clip(np.dot(cc, c), -1, 1))
            cr = max(d_a, d_b, d_c)
            
            max_cr_global = max(max_cr_global, cr)
            facet_data.append((simplex, cc, cr))
            
        if max_cr_global < 1e-6:
            break

        # Interpolate beta from beta_start to beta_end
        progress = it / max_iter
        beta = beta_start + (beta_end - beta_start) * progress
        
        for simplex, cc, cr in facet_data:
            w = math.exp(beta * (cr - max_cr_global))
            
            for idx in simplex:
                weights_sum[idx] += w
                vec_sum[idx] += w * cc
                
        moved = False
        for i in range(n):
            if weights_sum[i] > 1e-10:
                target = vec_sum[i] / weights_sum[i]
                norm_t = np.linalg.norm(target)
                if norm_t > 1e-10:
                    target = target / norm_t
                    
                    vec = target - pts[i]
                    vec_tan = vec - np.dot(vec, pts[i]) * pts[i]
                    norm_vec = np.linalg.norm(vec_tan)
                    
                    if norm_vec > 1e-10:
                        pts[i] = pts[i] + step_size * vec_tan
                        moved = True
                        
        if not moved:
            break
            
        pts = _normalize(pts)
        
        # Decay step size
        step_size *= decay
        
        if step_size < 1e-8:
            break

    return pts
=======
def _optimize_facet_shrink(pts, max_iter=1000, initial_step=0.5, decay=0.995, beta_start=2.0, beta_end=20.0, deadline=None):
    """
    Optimizes the configuration by moving each point towards the weighted
    average of the circumcenters of its incident facets.
    Uses adaptive step size and increasing beta to focus on worst facets.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    step_size = initial_step
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        weights_sum = np.zeros(n)
        vec_sum = np.zeros((n, 3))
        
        max_cr_global = 0.0
        facet_data = []

        for simplex in hull.simplices:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            v = np.cross(a - b, b - c)
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-12:
                continue
            u = v / norm_v
            
            n_face = np.cross(b - a, c - a)
            norm_n = np.linalg.norm(n_face)
            if norm_n < 1e-12:
                continue
            n_face = n_face / norm_n
            if np.dot(n_face, a) < 0:
                n_face = -n_face
                
            if np.dot(u, n_face) > 0:
                cc = u
            else:
                cc = -u
                
            d_a = np.arccos(np.clip(np.dot(cc, a), -1, 1))
            d_b = np.arccos(np.clip(np.dot(cc, b), -1, 1))
            d_c = np.arccos(np.clip(np.dot(cc, c), -1, 1))
            cr = max(d_a, d_b, d_c)
            
            max_cr_global = max(max_cr_global, cr)
            facet_data.append((simplex, cc, cr))
            
        if max_cr_global < 1e-6:
            break

        # Interpolate beta from beta_start to beta_end
        progress = it / max_iter
        beta = beta_start + (beta_end - beta_start) * progress
        
        # Use log-sum-exp for numerical stability and smoother weighting
        crs = np.array([cr for _, _, cr in facet_data])
        if len(crs) > 0:
            crs_shifted = beta * (crs - max_cr_global)
            max_shifted = np.max(crs_shifted)
            log_weights = crs_shifted - max_shifted
            exp_weights = np.exp(log_weights)
            sum_exp = np.sum(exp_weights)
            if sum_exp > 1e-30:
                normalized_weights = exp_weights / sum_exp
            else:
                normalized_weights = np.ones(len(crs)) / len(crs)
        else:
            normalized_weights = np.array([])
        
        for idx_facet, (simplex, cc, cr) in enumerate(facet_data):
            w = normalized_weights[idx_facet] if len(normalized_weights) > 0 else 0.0
            
            for idx in simplex:
                weights_sum[idx] += w
                vec_sum[idx] += w * cc
                
        moved = False
        for i in range(n):
            if weights_sum[i] > 1e-10:
                target = vec_sum[i] / weights_sum[i]
                norm_t = np.linalg.norm(target)
                if norm_t > 1e-10:
                    target = target / norm_t
                    
                    vec = target - pts[i]
                    vec_tan = vec - np.dot(vec, pts[i]) * pts[i]
                    norm_vec = np.linalg.norm(vec_tan)
                    
                    if norm_vec > 1e-10:
                        pts[i] = pts[i] + step_size * vec_tan
                        moved = True
                        
        if not moved:
            break
            
        pts = _normalize(pts)
        
        # Decay step size
        step_size *= decay
        
        if step_size < 1e-8:
            break

    return pts
>>>>>>> REPLACE

