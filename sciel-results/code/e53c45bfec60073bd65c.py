import numpy as np
import scipy.spatial
import time

N = 50

def _compute_soft_max_radius_and_grads(points, tau=0.1):
    """
    Computes the soft-max covering radius and gradients.
    Uses the convex hull facets. The soft-max is a smooth approximation of the max.
    """
    # Ensure points are on the sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Compute Convex Hull
    try:
        hull = scipy.spatial.ConvexHull(points)
    except Exception:
        # Fallback if hull fails
        return np.pi, np.zeros_like(points)

    n_facets = len(hull.simplices)
    if n_facets == 0:
        return np.pi, np.zeros_like(points)

    radii = np.zeros(n_facets)
    centers = np.zeros((n_facets, 3))
    cos_radii = np.zeros(n_facets)
    
    # Precompute facet data
    for i, simplex in enumerate(hull.simplices):
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        
        ab = np.dot(a, b)
        ac = np.dot(a, c)
        bc = np.dot(b, c)
        
        Gram = np.array([
            [1.0, ab, ac],
            [ab, 1.0, bc],
            [ac, bc, 1.0]
        ])
        
        try:
            m = np.linalg.solve(Gram, np.ones(3))
            c_dir = m[0]*a + m[1]*b + m[2]*c
            norm_c_dir = np.linalg.norm(c_dir)
            if norm_c_dir < 1e-10:
                radii[i] = 0.0
                centers[i] = 0.0
                cos_radii[i] = 1.0
                continue
            c_center = c_dir / norm_c_dir
            centers[i] = c_center
            # Radius is arccos(dot(c_center, a))
            cos_r = np.dot(c_center, a)
            cos_r = np.clip(cos_r, -1.0, 1.0)
            cos_radii[i] = cos_r
            radii[i] = np.arccos(cos_r)
        except np.linalg.LinAlgError:
            # Degenerate, use centroid approximation
            c_center = (a+b+c)
            norm_c = np.linalg.norm(c_center)
            if norm_c < 1e-10:
                radii[i] = 0.0
                centers[i] = 0.0
                cos_radii[i] = 1.0
                continue
            c_center = c_center / norm_c
            centers[i] = c_center
            cos_r = np.dot(c_center, a)
            cos_r = np.clip(cos_r, -1.0, 1.0)
            cos_radii[i] = cos_r
            radii[i] = np.arccos(cos_r)

    # Compute soft-max radius
    # R_soft = tau * log(sum(exp(R_i / tau)))
    # To avoid overflow, subtract max
    max_r = np.max(radii)
    if max_r == 0:
        return 0.0, np.zeros_like(points)
    
    exp_terms = np.exp((radii - max_r) / tau)
    sum_exp = np.sum(exp_terms)
    soft_max_r = max_r + tau * np.log(sum_exp)
    
    # Weights for gradients
    weights = exp_terms / sum_exp
    
    # Calculate Gradients
    # Gradient of soft-max R w.r.t point p_j:
    # sum_i w_i * (d R_i / d p_j)
    # R_i depends on p_j only if p_j is a vertex of facet i.
    # For facet i with vertices v1, v2, v3 and center C, radius R:
    # dR/dv_k (for k in 1..3) = C - cos(R)v_k  (This is the gradient of the angle, pointing towards C)
    # We want to MINIMIZE R, so we use the negative gradient as the descent direction?
    # No, the gradient points in the direction of INCREASE.
    # So we should move in direction -grad.
    # Wait, the code usually does: new_p = p - step * grad.
    # So if we return grad = dR/dp, the update p - step*grad moves to decrease R.
    # My derived gradient dR/dv = C - cos(R)v points towards C (decreasing distance).
    # So if I move p in direction (C - cos(R)v), the distance DECREASES.
    # So dR/dv should be the vector that, when SUBTRACTED, moves towards C?
    # Let's check:
    # If I want to decrease R, I move towards C.
    # Direction towards C is $u = C - \cos(R)v$.
    # If I set grad = -u, then p - step*grad = p + step*u, which moves towards C.
    # So the gradient of the objective R is actually $- (C - \cos(R)v) = \cos(R)v - C$.
    # Let's verify with the formula $\nabla_v \arccos(C \cdot v) = \frac{-C}{\sin R}$.
    # This is the gradient in $\mathbb{R}^3$. We need the tangent component.
    # Tangent component of $\frac{-C}{\sin R}$ is $\frac{-C - (-C \cdot v)v}{\sin R} = \frac{-C + \cos R v}{\sin R}$.
    # So $\nabla_{sphere} R = \frac{\cos R v - C}{\sin R}$.
    # This vector points AWAY from C (since $C - \cos R v$ points towards C).
    # So moving in direction $\nabla R$ INCREASES R.
    # Moving in direction $-\nabla R$ DECREASES R.
    # So we return grad = $\sum w_i \frac{\cos R_i v_{ik} - C_i}{\sin R_i}$.
    # And the update is $p \leftarrow p - \text{step} \cdot \text{grad}$.
    
    grads = np.zeros_like(points)
    
    for i in range(n_facets):
        w = weights[i]
        if w < 1e-12:
            continue
        r = radii[i]
        if r < 1e-10:
            continue
        sin_r = np.sin(r)
        if sin_r < 1e-10:
            continue
        
        C = centers[i]
        cos_r = cos_radii[i]
        
        simplex = hull.simplices[i]
        for k in range(3):
            p_idx = simplex[k]
            v = points[p_idx]
            # Gradient contribution: (cos_r * v - C) / sin_r
            # This points away from C. Subtracting it moves towards C.
            grads[p_idx] += w * (cos_r * v - C) / sin_r
            
    # Add a small repulsive force to prevent clustering
    # Force = sum_{j != i} (p_i - p_j) / dist^2 ?
    # This is O(N^2). N=50, so 2500 ops. Negligible.
    repulsion_strength = 0.01
    for i in range(N):
        for j in range(i+1, N):
            v_i = points[i]
            v_j = points[j]
            diff = v_i - v_j
            dist_sq = np.dot(diff, diff)
            if dist_sq < 1e-8:
                continue
            # Force on i is proportional to (v_i - v_j) / dist_sq
            # This pushes i away from j.
            force = diff / dist_sq
            grads[i] -= repulsion_strength * force # Wait, if we subtract grad, and grad has -force, we add force?
            # We want to ADD repulsion to the gradient so that p - step*grad moves them apart?
            # If grad has a component towards j, p - step*grad moves away from j.
            # Vector from j to i is (v_i - v_j).
            # So we want grad to have component in direction (v_i - v_j).
            # So grads[i] += repulsion * (v_i - v_j) / dist_sq.
            # And grads[j] -= repulsion * (v_i - v_j) / dist_sq.
            grads[i] += repulsion_strength * force
            grads[j] -= repulsion_strength * force

    return soft_max_r, grads

def _local_search(points, time_budget):
    """
    Local search using gradient descent on the soft-max covering radius.
    """
    current_points = points.copy()
    # Initial evaluation
    best_radius, _ = _compute_soft_max_radius_and_grads(current_points)
    best_points = current_points.copy()
    
    start_time = time.time()
    step_size = 0.05
    decay = 0.995
    min_step = 1e-6
    
    while time.time() - start_time < time_budget:
        radius, grads = _compute_soft_max_radius_and_grads(current_points)
        
        if radius < best_radius:
            best_radius = radius
            best_points = current_points.copy()
            
        # Normalize gradients to control step size
        grad_norm = np.linalg.norm(grads)
        if grad_norm < 1e-10:
            break
            
        grads = grads / grad_norm
        
        # Update
        new_points = current_points - step_size * grads
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        current_points = new_points
        step_size *= decay
        if step_size < min_step:
            break

    return best_points, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    best_points = None
    best_radius = np.inf
    
    # Seeds
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
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        seeds.append(pts)
        
    # 3. More random starts
    for seed_val in [1, 2, 3]:
        rng = np.random.default_rng(seed_val)
        pts = rng.normal(size=(N, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        seeds.append(pts)

    # Process seeds
    for i, seed_pts in enumerate(seeds):
        if time.time() > deadline - 5:
            break
            
        t_start = time.time()
        pts, rad = _local_search(seed_pts, time_budget=2.0)
        
        if rad < best_radius:
            best_radius = rad
            best_points = pts.copy()
            
    # If we have time left, refine the best point found
    if best_points is not None and time.time() < deadline - 5:
        t_start = time.time()
        budget = deadline - 5 - time.time()
        pts, rad = _local_search(best_points, time_budget=budget)
        if rad < best_radius:
            best_radius = rad
            best_points = pts

    if best_points is None:
        # Fallback
        best_points = seeds[0]
        
    return best_points
