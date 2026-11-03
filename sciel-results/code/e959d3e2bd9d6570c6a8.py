<<<<<<< SEARCH
def _optimize_facet_shrink(pts, max_iter=500, step_size=0.1):
    """
    Greedy optimization: Find the facet with the largest circumradius,
    and move its vertices towards the circumcenter of that facet.
    Used as a final polish step.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    for it in range(max_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        max_cr = 0.0
        max_facet = None
        max_circum_center = None
        
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
            n_face = n_face / np.linalg.norm(n_face)
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
            
            if cr > max_cr:
                max_cr = cr
                max_facet = simplex
                max_circum_center = cc
                
        if max_facet is None:
            break
            
        for idx in max_facet:
            p = pts[idx]
            vec = max_circum_center - p
            vec_tan = vec - np.dot(vec, p) * p
            norm_vec = np.linalg.norm(vec_tan)
            if norm_vec > 1e-10:
                step = step_size * min(1.0, max_cr)
                pts[idx] = p + step * (vec_tan / norm_vec)
                
        pts = _normalize(pts)
        
        if max_cr < 1e-6:
            break
            
    return pts
=======
def _compute_facet_circumradii(pts, hull):
    """Compute circumradii and circumcenters for all facets."""
    n = len(pts)
    n_facets = len(hull.simplices)
    circumradii = np.zeros(n_facets)
    circumcenters = np.zeros((n_facets, 3))
    
    for i, simplex in enumerate(hull.simplices):
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        v = np.cross(a - b, b - c)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-12:
            d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
            d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
            d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
            cr = max(d_ab, d_bc, d_ca) / 2.0
            circumradii[i] = cr
            # Approximate circumcenter as midpoint of longest edge
            if d_ab >= d_bc and d_ab >= d_ca:
                cc = (a + b) / 2.0
            elif d_bc >= d_ab and d_bc >= d_ca:
                cc = (b + c) / 2.0
            else:
                cc = (c + a) / 2.0
            circumcenters[i] = cc / np.linalg.norm(cc)
            continue
        
        u = v / norm_v
        n_face = np.cross(b - a, c - a)
        n_face = n_face / np.linalg.norm(n_face)
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
        circumradii[i] = cr
        circumcenters[i] = cc
        
    return circumradii, circumcenters

def _optimize_smooth_max(pts, max_iter=300, beta=20.0, step_size=0.05):
    """
    Smooth optimization using soft-max surrogate for covering radius.
    Uses log-sum-exp to create a smooth approximation of max(circumradii).
    Moves points towards weighted circumcenters of incident facets.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    for it in range(max_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        circumradii, circumcenters = _compute_facet_circumradii(pts, hull)
        
        # Soft-max: smooth approximation of max
        # log(sum(exp(beta * cr_i))) / beta
        shifted = beta * circumradii - np.max(beta * circumradii)
        log_sum_exp = np.max(beta * circumradii) / beta + np.log(np.sum(np.exp(shifted))) / beta
        
        # Weights: softmax(beta * circumradii)
        weights = np.exp(shifted) / np.sum(np.exp(shifted))
        
        # Map facets to points
        point_weights = np.zeros(n)
        point_vecs = np.zeros((n, 3))
        
        for i, simplex in enumerate(hull.simplices):
            w = weights[i]
            cc = circumcenters[i]
            for idx in simplex:
                point_weights[idx] += w
                point_vecs[idx] += w * cc
        
        # Update points
        max_move = 0.0
        for i in range(n):
            if point_weights[i] < 1e-10:
                continue
            target = point_vecs[i] / np.linalg.norm(point_vecs[i])
            p = pts[i]
            vec = target - p
            vec_tan = vec - np.dot(vec, p) * p
            norm_vec = np.linalg.norm(vec_tan)
            if norm_vec > 1e-10:
                step = step_size * min(1.0, log_sum_exp)
                pts[i] = p + step * (vec_tan / norm_vec)
                move = np.linalg.norm(pts[i] - p)
                max_move = max(max_move, move)
                
        pts = _normalize(pts)
        
        if max_move < 1e-10:
            break
            
    return pts

def _polish_high_precision(pts, max_iter=200, tol=1e-12):
    """
    Final high-precision polish: alternate between Lloyd's and small-step
    gradient descent on the smooth max surrogate with very tight tolerance.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    for it in range(max_iter):
        # First, run a few Lloyd iterations
        pts = _spherical_lloyd(pts, max_iter=10, tol=1e-10)
        
        # Then, do small-step smooth optimization
        pts = _optimize_smooth_max(pts, max_iter=20, beta=50.0, step_size=0.01)
        
        # Check convergence
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        circumradii, _ = _compute_facet_circumradii(pts, hull)
        max_cr = np.max(circumradii)
        
        if max_cr < tol:
            break
            
    return pts
>>>>>>> REPLACE

