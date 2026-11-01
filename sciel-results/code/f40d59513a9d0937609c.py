import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 50

def _normalize(points):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms

def _compute_facet_data(points):
    """
    Computes the circumradius and circumcenter for each facet of the convex hull.
    Returns:
        radii: array of shape (n_facets,)
        centers: array of shape (n_facets, 3)
        cos_radii: array of shape (n_facets,)
        simplices: array of facet vertex indices
    """
    points = _normalize(points)
    
    try:
        hull = scipy.spatial.ConvexHull(points)
    except Exception:
        # Fallback: return empty
        return np.array([]), np.zeros((0, 3)), np.array([]), np.array([], dtype=int)
    
    n_facets = len(hull.simplices)
    radii = np.zeros(n_facets)
    centers = np.zeros((n_facets, 3))
    cos_radii = np.ones(n_facets)
    
    for i, simplex in enumerate(hull.simplices):
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        
        cross_bc = np.cross(b, c)
        cross_ca = np.cross(c, a)
        cross_ab = np.cross(a, b)
        
        numerator = cross_bc + cross_ca + cross_ab
        denominator = 2.0 * np.dot(a, cross_bc)
        
        if abs(denominator) < 1e-10:
            # Degenerate facet
            centers[i] = (a + b + c) / 3.0
            centers[i] = _normalize(centers[i].reshape(1, 3))[0]
            cos_radii[i] = np.dot(centers[i], a)
            radii[i] = np.arccos(np.clip(cos_radii[i], -1.0, 1.0))
            continue
            
        c_center = numerator / denominator
        c_norm = np.linalg.norm(c_center)
        
        if c_norm < 1e-10:
            centers[i] = (a + b + c) / 3.0
            centers[i] = _normalize(centers[i].reshape(1, 3))[0]
            cos_radii[i] = np.dot(centers[i], a)
            radii[i] = np.arccos(np.clip(cos_radii[i], -1.0, 1.0))
            continue
            
        centers[i] = c_center / c_norm
        cos_r = np.dot(centers[i], a)
        cos_r = np.clip(cos_r, -1.0, 1.0)
        cos_radii[i] = cos_r
        radii[i] = np.arccos(cos_r)
        
    return radii, centers, cos_radii, hull.simplices

def _compute_covering_radius(points):
    """Compute the exact covering radius (max facet circumradius)."""
    radii, _, _, _ = _compute_facet_data(points)
    if len(radii) == 0:
        return np.pi
    return np.max(radii)

def _soft_max_radius_and_grad(points, tau=0.05):
    """
    Compute the soft-max covering radius and its gradient.
    The gradient is computed correctly on the sphere.
    """
    points = _normalize(points)
    radii, centers, cos_radii, simplices = _compute_facet_data(points)
    
    if len(radii) == 0:
        return np.pi, np.zeros_like(points)
    
    # Soft max
    max_r = np.max(radii)
    if max_r < 1e-10:
        return 0.0, np.zeros_like(points)
        
    # Avoid overflow
    shifted = (radii - max_r) / tau
    # Clip to avoid underflow issues in exp
    shifted = np.clip(shifted, -50, 50)
    
    exp_terms = np.exp(shifted)
    sum_exp = np.sum(exp_terms)
    
    if sum_exp < 1e-15:
        return max_r, np.zeros_like(points)
        
    soft_max_r = max_r + tau * np.log(sum_exp)
    weights = exp_terms / sum_exp
    
    # Gradient
    grads = np.zeros_like(points)
    
    for i in range(len(radii)):
        w = weights[i]
        if w < 1e-8:
            continue
            
        r = radii[i]
        if r < 1e-8:
            continue
            
        sin_r = np.sin(r)
        if sin_r < 1e-8:
            continue
            
        C = centers[i]
        cos_r = cos_radii[i]
        
        simplex = simplices[i]
        for k in range(3):
            p_idx = simplex[k]
            v = points[p_idx]
            # Gradient of arccos(C dot v) w.r.t v on the sphere
            # d/dv arccos(C dot v) = -C / sin(R)
            # Tangential component: (-C + (C dot v) v) / sin(R) = (-C + cos(R) v) / sin(R)
            grads[p_idx] += w * (cos_r * v - C) / sin_r
            
    return soft_max_r, grads

def _lloyd_iteration(points, alpha=0.5, n_iters=20):
    """
    Lloyd-style iteration: move points toward their worst facet circumcenters.
    This helps equalize the covering radii across facets.
    """
    points = _normalize(points)
    best_points = points.copy()
    best_val = _compute_covering_radius(points)
    
    for it in range(n_iters):
        radii, centers, cos_radii, simplices = _compute_facet_data(points)
        if len(radii) == 0:
            break
        
        # For each point, find the worst facet containing it
        # and move it toward that facet's circumcenter
        new_points = points.copy()
        
        # Weight by facet radius (higher weight for worse facets)
        # Use softmax weighting to emphasize worst facets
        max_r = np.max(radii)
        if max_r > 1e-10:
            shifted = (radii - max_r) / 0.1
            shifted = np.clip(shifted, -50, 50)
            exp_terms = np.exp(shifted)
            sum_exp = np.sum(exp_terms)
            if sum_exp > 1e-15:
                facet_weights = exp_terms / sum_exp
            else:
                facet_weights = np.ones_like(radii) / len(radii)
        else:
            facet_weights = np.ones_like(radii) / len(radii)
        
        # Accumulate movement for each point
        point_moves = np.zeros_like(points)
        point_counts = np.zeros(N)
        
        for i in range(len(radii)):
            w = facet_weights[i]
            if w < 1e-6:
                continue
            
            simplex = simplices[i]
            C = centers[i]
            
            for k in range(3):
                p_idx = simplex[k]
                point_moves[p_idx] += w * (C - points[p_idx])
                point_counts[p_idx] += w
        
        # Apply moves with adaptive step size
        for p_idx in range(N):
            if point_counts[p_idx] > 1e-6:
                move = point_moves[p_idx] / point_counts[p_idx]
                move_norm = np.linalg.norm(move)
                if move_norm > 1e-10:
                    # Scale move to be a fraction of the distance
                    step = alpha * min(1.0, move_norm / max_r if max_r > 1e-10 else 1.0)
                    new_points[p_idx] = points[p_idx] + step * move
        
        new_points = _normalize(new_points)
        val = _compute_covering_radius(new_points)
        
        if val < best_val:
            best_val = val
            best_points = new_points.copy()
            points = new_points
        else:
            # If no improvement, reduce alpha and try again
            alpha *= 0.8
            if alpha < 0.01:
                break
    
    return best_points, best_val

def _lbfgs_stage(points, time_budget):
    """
    Optimize points using L-BFGS-B on the soft-max radius.
    """
    points = _normalize(points)
    start_time = time.time()
    
    def objective(x):
        if time.time() - start_time > time_budget:
            return np.pi
        pts = _normalize(x.reshape(N, 3))
        val, _ = _soft_max_radius_and_grad(pts)
        return val
        
    def gradient(x):
        if time.time() - start_time > time_budget:
            return np.zeros_like(x)
        pts = _normalize(x.reshape(N, 3))
        _, grad = _soft_max_radius_and_grad(pts)
        return grad.flatten()
        
    try:
        res = scipy.optimize.minimize(
            objective, 
            points.flatten(), 
            jac=gradient, 
            method='L-BFGS-B',
            options={'maxiter': 300, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if res.success or res.fun < np.pi:
            return _normalize(res.x.reshape(N, 3)), res.fun
    except Exception:
        pass
        
    return points, np.pi

def _local_polish(points, time_budget):
    """
    Fine-tune the points using a steepest descent with line search on the sphere.
    """
    points = _normalize(points)
    start_time = time.time()
    best_points = points.copy()
    best_val = _compute_covering_radius(points)
    
    step_size = 0.01
    min_step = 1e-10
    
    while time.time() - start_time < time_budget:
        val, grad = _soft_max_radius_and_grad(points, tau=0.01)
        
        if val < best_val:
            best_val = val
            best_points = points.copy()
            
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
            
        # Normalize gradient
        grad = grad / grad_norm
        
        # Line search with backtracking
        best_step = step_size
        best_local_val = val
        
        for _ in range(20):
            new_points = points - best_step * grad
            new_points = _normalize(new_points)
            new_val, _ = _soft_max_radius_and_grad(new_points, tau=0.01)
            
            if new_val < best_local_val:
                best_local_val = new_val
                step_size = best_step * 1.5
            else:
                best_step = best_step * 0.5
                
            if best_step < min_step:
                break
                
        points = _normalize(points - best_step * grad)
        
        if best_step < min_step:
            break
            
    return best_points, best_val

def _generate_seeds():
    """Generate initial configurations."""
    seeds = []
    
    # 1. Fibonacci Sphere
    j = np.arange(N) + 0.5
    z = 1.0 - 2.0 * j / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * j
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    seeds.append(fib)
    
    # 2. Random starts
    rng = np.random.default_rng(42)
    for _ in range(15):
        pts = rng.normal(size=(N, 3))
        pts = _normalize(pts)
        seeds.append(pts)
        
    # 3. More random starts with different seeds
    for seed_val in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        rng = np.random.default_rng(seed_val)
        pts = rng.normal(size=(N, 3))
        pts = _normalize(pts)
        seeds.append(pts)
        
    return seeds

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_points = None
    best_radius = np.inf
    
    seeds = _generate_seeds()
    
    # Process seeds with multiple optimization stages
    for i, seed_pts in enumerate(seeds):
        if time.time() > deadline - 15:
            break
        
        current = seed_pts.copy()
        
        # Stage 1: Lloyd iteration for fast equalization
        if time.time() < deadline - 10:
            current, rad = _lloyd_iteration(current, alpha=0.3, n_iters=30)
            if rad < best_radius:
                best_radius = rad
                best_points = current.copy()
        
        # Stage 2: L-BFGS for smooth optimization
        if time.time() < deadline - 8:
            current, rad = _lbfgs_stage(current, time_budget=3.0)
            if rad < best_radius:
                best_radius = rad
                best_points = current.copy()
        
        # Stage 3: Another Lloyd pass to equalize
        if best_points is not None and time.time() < deadline - 5:
            current, rad = _lloyd_iteration(best_points, alpha=0.1, n_iters=50)
            if rad < best_radius:
                best_radius = rad
                best_points = current.copy()
        
        # Stage 4: Local polish
        if best_points is not None and time.time() < deadline - 2:
            current, rad = _local_polish(best_points, time_budget=2.0)
            if rad < best_radius:
                best_radius = rad
                best_points = current.copy()
    
    # Final polish with remaining time
    if best_points is not None and time.time() < deadline - 1:
        remaining = deadline - 1 - time.time()
        if remaining > 1.0:
            # Do a final Lloyd + polish cycle
            pts, rad = _lloyd_iteration(best_points, alpha=0.05, n_iters=100)
            if rad < best_radius:
                best_radius = rad
                best_points = pts.copy()
            
            if time.time() < deadline - 0.5:
                remaining2 = deadline - 0.5 - time.time()
                if remaining2 > 0.5:
                    pts, rad = _local_polish(best_points, time_budget=remaining2)
                    if rad < best_radius:
                        best_radius = rad
                        best_points = pts

    if best_points is None:
        best_points = seeds[0]
        
    return _normalize(best_points)
