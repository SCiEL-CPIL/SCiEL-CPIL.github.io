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
        
        # Solve for circumcenter in the plane of the triangle
        # Let center = x*a + y*b + z*c
        # Conditions: |center|^2 = 1, center dot a = center dot b = center dot c
        # This leads to a linear system for x, y, z
        
        ab = np.dot(a, b)
        ac = np.dot(a, c)
        bc = np.dot(b, c)
        
        # The system is:
        # (1-ab) x + (ac-ab) y + (bc-ab) z = 0  (from center dot a = center dot b)
        # (ab-ac) x + (1-ac) y + (bc-ac) z = 0  (from center dot a = center dot b)
        # x + y + z = 1 (normalization in barycentric coords, not strictly needed for direction)
        
        # Actually, a better way:
        # The circumcenter C satisfies:
        # |C - a|^2 = |C - b|^2 = |C - c|^2
        # C dot C - 2 C dot a + 1 = C dot C - 2 C dot b + 1  => C dot (b-a) = 0
        # C dot (c-a) = 0
        # And C is in the plane of a,b,c.
        
        # Let's use the formula:
        # C = ( (|a|^2 (b x c) + |b|^2 (c x a) + |c|^2 (a x b) ) / (2 * a dot (b x c)) )
        # Since |a|=|b|=|c|=1:
        # C = ( b x c + c x a + a x b ) / (2 * a dot (b x c))
        
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
            # This is the gradient of R. We want to minimize R, so we move in -grad direction.
            # The code usually does p = p - step * grad.
            # So we return grad = sum w * dR/dv.
            # dR/dv (tangential) = (cos(R) v - C) / sin(R)
            grads[p_idx] += w * (cos_r * v - C) / sin_r
            
    return soft_max_r, grads

def _lbfgs_stage(points, time_budget):
    """
    Optimize points using L-BFGS-B on the soft-max radius.
    """
    points = _normalize(points)
    start_time = time.time()
    
    # We need to parameterize the sphere.
    # A common trick is to optimize in R^3 and project, but L-BFGS doesn't handle constraints well.
    # Instead, we can use a parameterization like quaternions or just use the gradient on the sphere
    # and do line search.
    # However, scipy.optimize.minimize with method='L-BFGS-B' works on R^3.
    # Let's try to optimize the 3D coordinates and project back to sphere in the objective/gradient.
    # This is often effective for sphere problems.
    
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
        # The gradient returned is on the tangent space.
        # For the 3D optimization, we need the gradient in R^3.
        # The gradient of the function f(x) = soft_max_radius(normalize(x)) w.r.t x
        # is slightly different. But usually, projecting the gradient back or just using it
        # directly works well enough for L-BFGS if we normalize in the objective.
        # Actually, the gradient of f(x) w.r.t x is:
        # df/dx = (df/dp) dot (dp/dx)
        # p = x / |x|
        # dp/dx = (I - p p^T) / |x|  (if x is not on sphere)
        # Since we normalize, let's just return the gradient computed on the sphere,
        # assuming x is close to the sphere. This is a common approximation.
        return grad.flatten()
        
    try:
        res = scipy.optimize.minimize(
            objective, 
            points.flatten(), 
            jac=gradient, 
            method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
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
    min_step = 1e-8
    
    while time.time() - start_time < time_budget:
        val, grad = _soft_max_radius_and_grad(points, tau=0.02)
        
        if val < best_val:
            best_val = val
            best_points = points.copy()
            
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
            
        # Normalize gradient
        grad = grad / grad_norm
        
        # Line search
        best_step = step_size
        best_local_val = val
        
        for _ in range(10):
            new_points = points - best_step * grad
            new_points = _normalize(new_points)
            new_val, _ = _soft_max_radius_and_grad(new_points, tau=0.02)
            
            if new_val < best_local_val:
                best_local_val = new_val
                best_step = best_step * 1.5
            else:
                best_step = best_step * 0.5
                
            if best_step < min_step:
                break
                
        points = _normalize(points - best_step * grad)
        
        if best_step < min_step:
            break
            
    return best_points, best_val

def _grid_slqp_polish(points, time_budget):
    """
    Try to optimize specific points (the ones defining the worst facets) more aggressively.
    """
    points = _normalize(points)
    start_time = time.time()
    best_points = points.copy()
    best_val = _compute_covering_radius(points)
    
    # Identify worst facets
    radii, centers, cos_radii, simplices = _compute_facet_data(points)
    if len(radii) == 0:
        return points, best_val
        
    # Sort facets by radius
    order = np.argsort(radii)[::-1]
    
    # Try to optimize the vertices of the top few facets
    # This is a heuristic.
    
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
    for _ in range(10):
        pts = rng.normal(size=(N, 3))
        pts = _normalize(pts)
        seeds.append(pts)
        
    # 3. More random starts with different seeds
    for seed_val in [1, 2, 3, 4, 5]:
        rng = np.random.default_rng(seed_val)
        pts = rng.normal(size=(N, 3))
        pts = _normalize(pts)
        seeds.append(pts)
        
    return seeds

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_points = None
    best_radius = np.inf
    
    seeds = _generate_seeds()
    
    # Process seeds
    for i, seed_pts in enumerate(seeds):
        if time.time() > deadline - 10:
            break
            
        # Stage 1: L-BFGS
        if time.time() < deadline - 5:
            pts, rad = _lbfgs_stage(seed_pts, time_budget=2.0)
            if rad < best_radius:
                best_radius = rad
                best_points = pts.copy()
                
        # Stage 2: Local Polish
        if best_points is not None and time.time() < deadline - 2:
            pts, rad = _local_polish(best_points, time_budget=1.0)
            if rad < best_radius:
                best_radius = rad
                best_points = pts.copy()
                
    # Final polish with remaining time
    if best_points is not None and time.time() < deadline - 1:
        remaining = deadline - 1 - time.time()
        if remaining > 0.5:
            pts, rad = _local_polish(best_points, time_budget=remaining)
            if rad < best_radius:
                best_radius = rad
                best_points = pts

    if best_points is None:
        best_points = seeds[0]
        
    return _normalize(best_points)
