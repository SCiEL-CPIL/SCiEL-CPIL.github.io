import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50
TIME_LIMIT = 140.0  # safety margin from 150s

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _compute_facet_data(points):
    """
    Compute convex hull and facet circumcenters/radii (vectorized).
    Returns: hull, centers (array), radii (array), worst_idx
    """
    points = _normalize(points)
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return None, None, None, None
    
    n_facets = len(hull.simplices)
    if n_facets == 0:
        return hull, np.zeros((0, 3)), np.zeros(0), None
    
    # Get all facet vertices
    a = points[hull.simplices[:, 0]]
    b = points[hull.simplices[:, 1]]
    c = points[hull.simplices[:, 2]]
    
    # Compute dot products (vectorized)
    ab = np.sum(a * b, axis=1)
    ac = np.sum(a * c, axis=1)
    bc = np.sum(b * c, axis=1)
    
    # Build Gram matrices and solve (vectorized via batch solve)
    # Gram[i] = [[1, ab_i, ac_i], [ab_i, 1, bc_i], [ac_i, bc_i, 1]]
    ones = np.ones((n_facets, 3))
    
    # Solve using the formula for circumcenter of spherical triangle
    # m = Gram^{-1} * [1,1,1]^T
    # We can use the explicit inverse for 3x3 with structure
    # Or just use np.linalg.solve in a loop (faster for small n)
    
    centers = np.zeros((n_facets, 3))
    radii = np.zeros(n_facets)
    
    # Vectorized solve: for each facet, solve 3x3 system
    # Use the closed-form solution for circumcenter
    for i in range(n_facets):
        G = np.array([[1.0, ab[i], ac[i]],
                      [ab[i], 1.0, bc[i]],
                      [ac[i], bc[i], 1.0]])
        try:
            m = np.linalg.solve(G, ones[i])
            c_dir = m[0]*a[i] + m[1]*b[i] + m[2]*c[i]
            norm_c_dir = np.linalg.norm(c_dir)
            if norm_c_dir < 1e-10:
                continue
            centers[i] = c_dir / norm_c_dir
            radii[i] = np.arccos(np.clip(np.dot(centers[i], a[i]), -1.0, 1.0))
        except np.linalg.LinAlgError:
            continue
    
    if n_facets > 0 and np.any(radii > 0):
        worst_idx = np.argmax(radii)
    else:
        worst_idx = None
    
    return hull, centers, radii, worst_idx

def _compute_covering_radius(points):
    """
    Compute the actual covering radius (max facet circumradius).
    """
    hull, centers, radii, worst_idx = _compute_facet_data(points)
    if hull is None or worst_idx is None:
        return np.pi
    return radii[worst_idx]

def _soft_max_radius_and_grad(points, beta=50.0):
    """
    Compute soft-max radius and its gradient w.r.t. points.
    Optimized: precompute all facet data once, use vectorized operations where possible.
    """
    hull, centers, radii, worst_idx = _compute_facet_data(points)
    if hull is None or worst_idx is None:
        return np.pi, np.zeros_like(points)
    
    n_facets = len(hull.simplices)
    
    # Numerically stable log-sum-exp
    max_r = np.max(radii)
    shifted = beta * (radii - max_r)
    max_shifted = np.max(shifted)
    exp_terms = np.exp(shifted - max_shifted)
    sum_exp = np.sum(exp_terms)
    log_sum = max_shifted + np.log(sum_exp)
    soft_radius = max_r + log_sum / beta
    weights = exp_terms / sum_exp
    
    grads = np.zeros_like(points)
    
    # Vectorized gradient computation
    # For each facet i with weight w_i:
    #   grad_contrib_i = -centers[i] / sin(radii[i])
    #   For each vertex k of facet i:
    #     p = points[facet_vertex_k]
    #     tangential = grad_contrib_i - dot(grad_contrib_i, p) * p
    #     grads[facet_vertex_k] += w_i * tangential
    
    valid = (weights > 1e-15) & (radii > 1e-12)
    if not np.any(valid):
        return soft_radius, grads
    
    # Get data for valid facets only
    w_valid = weights[valid]
    C_valid = centers[valid]
    r_valid = radii[valid]
    sin_r_valid = np.sin(r_valid)
    sin_r_valid = np.maximum(sin_r_valid, 1e-10)
    
    grad_contrib = -C_valid / sin_r_valid[:, np.newaxis]  # (n_valid, 3)
    
    # Get vertex indices for valid facets
    valid_indices = np.where(valid)[0]
    simplices_valid = hull.simplices[valid_indices]  # (n_valid, 3)
    
    # For each valid facet and each of its 3 vertices, accumulate gradient
    for i in range(len(valid_indices)):
        w = w_valid[i]
        gc = grad_contrib[i]
        for k in range(3):
            p_idx = simplices_valid[i, k]
            p = points[p_idx]
            radial = np.dot(gc, p) * p
            tangential = gc - radial
            grads[p_idx] += w * tangential
    
    return soft_radius, grads

def _project_to_sphere(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _repulsive_relaxation(points, time_budget, iterations=200):
    """
    Vectorized repulsion-based relaxation to improve initial spread.
    Uses adaptive step size with backtracking.
    """
    start_time = time.time()
    pts = points.copy()
    best_pts = pts.copy()
    best_radius = _compute_covering_radius(pts)
    step = 0.008
    
    for _ in range(iterations):
        if time.time() - start_time > time_budget:
            break
        
        # Vectorized pairwise repulsion
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        
        # Force: diff / dist^3
        dist_cubed = dist_sq * np.sqrt(dist_sq)
        forces_raw = diff / dist_cubed[:, :, np.newaxis]
        
        # Project to tangent space
        radial_dots = np.sum(forces_raw * pts[:, np.newaxis, :], axis=2)
        forces_tangent = forces_raw - radial_dots[:, :, np.newaxis] * pts[:, np.newaxis, :]
        
        # Sum over all j for each i
        forces = np.sum(forces_tangent, axis=1)
        
        # Normalize forces
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms = np.maximum(force_norms, 1e-15)
        normalized_forces = forces / force_norms
        
        # Backtracking line search
        alpha = step
        accepted = False
        for _ in range(8):
            new_pts = pts + alpha * normalized_forces
            new_pts = _project_to_sphere(new_pts)
            new_radius = _compute_covering_radius(new_pts)
            
            if new_radius < best_radius - 1e-15:
                best_radius = new_radius
                best_pts = new_pts.copy()
                pts = new_pts
                step = min(alpha * 1.3, 0.05)
                accepted = True
                break
            alpha *= 0.5
        
        if not accepted:
            step *= 0.7
            if step < 1e-7:
                break
    
    return best_pts, best_radius

def _lbfgs_stage(points, time_budget):
    """
    Use L-BFGS-B to minimize the soft-max covering radius.
    Uses progressive beta stages with tighter tolerances.
    """
    start_time = time.time()
    best_pts = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Progressive beta stages: start smooth, end sharp
    betas = [15.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0]
    
    for beta in betas:
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 2.0:
            break
        
        # Allocate more iterations for later (sharper) betas
        if beta <= 60:
            maxiter = 300
            maxfun = 2000
        elif beta <= 240:
            maxiter = 400
            maxfun = 3000
        else:
            maxiter = 500
            maxfun = 4000
        
        def objective_and_grad(x, _beta=beta):
            if time.time() - start_time > time_budget:
                raise Exception("timeout")
            pts = _project_to_sphere(x.reshape(N, 3))
            soft_r, grads = _soft_max_radius_and_grad(pts, beta=_beta)
            grad_flat = grads.reshape(-1)
            return soft_r, grad_flat
        
        x0 = best_pts.reshape(-1)
        
        try:
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': maxiter, 'maxfun': maxfun, 
                         'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            new_pts = _project_to_sphere(result.x.reshape(N, 3))
            new_radius = _compute_covering_radius(new_pts)
            if new_radius < best_radius - 1e-13:
                best_radius = new_radius
                best_pts = new_pts
        except Exception:
            pass
    
    return best_pts

def _local_polish(points, time_budget, beta=100.0):
    """
    Final local polish using projected gradient descent on tangent space.
    Directly minimizes the soft-max covering radius.
    Uses adaptive step size with line search and adaptive beta.
    """
    current_points = points.copy()
    best_radius = _compute_covering_radius(current_points)
    best_points = current_points.copy()
    
    start_time = time.time()
    step_size = 0.015
    min_step = 1e-15
    decay = 0.75
    stagnant = 0
    
    # Adaptive beta: start moderate, increase as we converge
    current_beta = beta
    min_improvement = 1e-14
    
    while time.time() - start_time < time_budget:
        soft_radius, grads = _soft_max_radius_and_grad(current_points, beta=current_beta)
        
        true_radius = _compute_covering_radius(current_points)
        if true_radius < best_radius - min_improvement:
            best_radius = true_radius
            best_points = current_points.copy()
            stagnant = 0
            # Increase beta as we get closer to optimum
            if true_radius < np.radians(18.31):
                current_beta = min(current_beta * 1.3, 2000.0)
        else:
            stagnant += 1
        
        if stagnant > 60:
            break
        
        grad_norm = np.linalg.norm(grads)
        if grad_norm < 1e-16:
            break
        
        direction = -grads / grad_norm
        
        # Backtracking line search with more attempts
        alpha = step_size
        accepted = False
        for _ in range(12):
            new_points = current_points + alpha * direction
            new_points = _project_to_sphere(new_points)
            new_radius = _compute_covering_radius(new_points)
            
            if new_radius < true_radius - min_improvement:
                current_points = new_points
                step_size = min(alpha * 1.4, 0.12)
                accepted = True
                break
            alpha *= 0.45
        
        if not accepted:
            step_size *= decay
            if step_size < min_step:
                break
    
    return best_points, best_radius

def _worst_facet_optimization(points, step_size, time_budget, iterations=200):
    """
    Iteratively move the vertices of the worst facet towards the worst center.
    Uses adaptive step size and tangent space projection.
    """
    start_time = time.time()
    pts = points.copy()
    prev_radius = np.pi
    
    for _ in range(iterations):
        if time.time() - start_time > time_budget:
            break
        
        hull, centers, radii, worst_idx = _compute_facet_data(pts)
        if hull is None or worst_idx is None:
            break
        worst_center = centers[worst_idx]
        radius = radii[worst_idx]
        
        if radius >= prev_radius:
            step_size *= 0.5
            if step_size < 1e-8:
                break
        else:
            prev_radius = radius
        
        # Find facets that are close to the worst center
        target_indices_set = set()
        for i, simplex in enumerate(hull.simplices):
            c = centers[i]
            if np.dot(c, worst_center) > 0.999:
                for idx in simplex:
                    target_indices_set.add(idx)
        
        if not target_indices_set:
            break
        
        # Move vertices towards worst_center in tangent space
        for idx in target_indices_set:
            v = pts[idx]
            dir_vec = worst_center - v
            radial_comp = np.dot(dir_vec, v)
            dir_vec = dir_vec - radial_comp * v
            norm_dir = np.linalg.norm(dir_vec)
            if norm_dir > 1e-10:
                dir_vec /= norm_dir
                pts[idx] = v + step_size * dir_vec
                pts[idx] /= np.linalg.norm(pts[idx])
    
    return pts

def _hull_volume_and_gradient(points):
    """
    Compute the convex hull volume and its gradient w.r.t. point positions.
    The volume is a sum of tetrahedra from origin to each facet.
    Volume = (1/6) * sum over facets of det([a,b,c]) where a,b,c are facet vertices.
    Gradient w.r.t. point i: for each facet containing i, the contribution is
    (1/6) * cross(other1, other2) (with appropriate sign based on orientation).
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return 0.0, np.zeros_like(points)
    
    n_facets = len(hull.simplices)
    grad = np.zeros_like(points)
    volume = 0.0
    
    for simplex in hull.simplices:
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        
        # Volume contribution: (1/6) |det([a,b,c])|
        # Since hull is outward-facing, det should be positive for outward normal
        det_val = np.dot(a, np.cross(b, c))
        volume += det_val / 6.0
        
        # Gradient: d/d(points[i]) det([a,b,c])
        # d/da det(a,b,c) = cross(b,c)
        # d/db det(a,b,c) = cross(c,a)
        # d/dc det(a,b,c) = cross(a,b)
        grad[simplex[0]] += np.cross(b, c) / 6.0
        grad[simplex[1]] += np.cross(c, a) / 6.0
        grad[simplex[2]] += np.cross(a, b) / 6.0
    
    return volume, grad

def _pressure_relaxation(points, time_budget, iterations=100):
    """
    Pressure-minimization relaxation: move points along the negative gradient
    of hull volume projected to tangent space. This pushes points to maximize
    the hull volume, spreading them out and reducing covering radius.
    """
    start_time = time.time()
    pts = points.copy()
    best_pts = pts.copy()
    best_radius = _compute_covering_radius(pts)
    step = 0.01
    
    for _ in range(iterations):
        if time.time() - start_time > time_budget:
            break
        
        vol, grad = _hull_volume_and_gradient(pts)
        
        # Project gradient to tangent space
        radial = np.sum(grad * pts, axis=1, keepdims=True)
        grad_tangent = grad - radial * pts
        
        # Normalize for consistent step size
        grad_norms = np.linalg.norm(grad_tangent, axis=1, keepdims=True)
        grad_norms = np.maximum(grad_norms, 1e-15)
        grad_unit = grad_tangent / grad_norms
        
        # Move in direction of increasing volume (positive gradient)
        new_pts = pts + step * grad_unit
        new_pts = _project_to_sphere(new_pts)
        
        new_radius = _compute_covering_radius(new_pts)
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            step *= 0.7
            if step < 1e-7:
                break
    
    return best_pts, best_radius

def _lloyd_iteration(points, time_budget, iterations=50):
    """
    Lloyd's algorithm for spherical k-means as a local refinement step.
    """
    start_time = time.time()
    pts = points.copy()
    best_pts = pts.copy()
    best_radius = _compute_covering_radius(pts)
    
    for _ in range(iterations):
        if time.time() - start_time > time_budget:
            break
        
        dots = pts @ pts.T
        new_pts = np.zeros_like(pts)
        
        for i in range(N):
            dists = np.arccos(np.clip(dots[i], -1.0, 1.0))
            mask = dists < 0.5
            if np.sum(mask) > 1:
                neighbors = pts[mask]
                centroid = np.sum(neighbors, axis=0)
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm > 1e-10:
                    centroid /= centroid_norm
                    new_pts[i] = pts[i] * 0.7 + centroid * 0.3
                else:
                    new_pts[i] = pts[i]
            else:
                new_pts[i] = pts[i]
        
        new_pts = _project_to_sphere(new_pts)
        new_radius = _compute_covering_radius(new_pts)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            break
    
    return best_pts, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_points = None
    best_radius = np.inf
    
    # Generate diverse seeds
    seeds = []
    
    # 1. Fibonacci sphere
    j = np.arange(N) + 0.5
    z = 1.0 - 2.0 * j / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * j
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    seeds.append(fib)
    
    # 2. Random starts with different seeds
    for seed_val in [42, 1, 2, 3, 7, 13, 99, 123, 555, 777]:
        rng = np.random.default_rng(seed_val)
        pts = rng.normal(size=(N, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        seeds.append(pts)
    
    # 3. Icosahedron-based
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
    rng_ico = np.random.default_rng(12345)
    extra = rng_ico.normal(size=(50 - len(ico_verts), 3))
    extra /= np.linalg.norm(extra, axis=1, keepdims=True)
    ico_seed = np.vstack([ico_verts, extra])
    seeds.append(ico_seed)
    
    n_seeds = len(seeds)
    
    # Stage 1: For each seed, apply repulsion relaxation, then L-BFGS, then local polish
    stage1_deadline = deadline - 35.0
    
    for i, seed_pts in enumerate(seeds):
        elapsed = time.time() - start_time
        remaining = stage1_deadline - elapsed
        
        if remaining < 5:
            break
        
        if i < n_seeds - 1:
            time_budget = min(8.0, remaining / (n_seeds - i))
        else:
            time_budget = min(12.0, remaining)
        
        if time_budget < 3.0:
            break
        
        # Step 1: Repulsion relaxation for initial spread
        elapsed = time.time() - start_time
        remaining = stage1_deadline - elapsed
        if remaining > 3:
            pts, rad = _repulsive_relaxation(seed_pts, min(2.0, remaining - 2))
        else:
            pts = seed_pts
            rad = _compute_covering_radius(pts)
        
        # Step 1.5: Pressure relaxation (volume-maximizing)
        elapsed = time.time() - start_time
        remaining = stage1_deadline - elapsed
        if remaining > 3:
            pts, rad = _pressure_relaxation(pts, min(1.5, remaining - 2))
        
        # Step 2: L-BFGS optimization with progressive beta
        elapsed = time.time() - start_time
        remaining = stage1_deadline - elapsed
        if remaining > 3:
            pts = _lbfgs_stage(pts, min(4.0, remaining - 2))
        
        # Step 3: Local polish
        elapsed = time.time() - start_time
        remaining = stage1_deadline - elapsed
        if remaining > 3:
            pts, rad = _local_polish(pts, min(3.0, remaining - 2), beta=100.0)
        else:
            rad = _compute_covering_radius(pts)
        
        if rad < best_radius:
            best_radius = rad
            best_points = pts.copy()
    
    # Stage 2: Final polish on best candidate
    if best_points is not None:
        # Pressure relaxation on best candidate
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 12:
            pts, rad = _pressure_relaxation(best_points, min(3.0, remaining - 3), iterations=80)
            if rad < best_radius:
                best_radius = rad
                best_points = pts.copy()
        
        # Worst-facet optimization with adaptive step sizes
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 12:
            for step in [0.05, 0.02, 0.01, 0.005, 0.002]:
                elapsed = time.time() - start_time
                remaining = deadline - elapsed
                if remaining < 5:
                    break
                pts = _worst_facet_optimization(best_points, step, min(3.0, remaining - 3), iterations=80)
                rad = _compute_covering_radius(pts)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Lloyd's iteration for local refinement
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 10:
            pts, rad = _lloyd_iteration(best_points, min(5.0, remaining - 5), iterations=50)
            if rad < best_radius:
                best_radius = rad
                best_points = pts.copy()
        
        # Fine local polish with moderate beta
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 10:
            budget = min(7.0, remaining - 5)
            if budget > 4:
                pts, rad = _local_polish(best_points, budget, beta=150.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Very fine polish with high beta for final convergence
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 8:
            budget = min(5.0, remaining - 4)
            if budget > 3:
                pts, rad = _local_polish(best_points, budget, beta=400.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Ultra-fine polish with very high beta
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 6:
            budget = min(4.0, remaining - 3)
            if budget > 2:
                pts, rad = _local_polish(best_points, budget, beta=800.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Final ultra-precise polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 4:
            budget = min(3.0, remaining - 2)
            if budget > 1.5:
                pts, rad = _local_polish(best_points, budget, beta=1500.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
    
    if best_points is None:
        best_points = seeds[0]
    
    # Ensure output is normalized and valid
    best_points = _normalize(best_points)
    
    # Final sanity check: verify no coincident points, not in hemisphere
    norms = np.linalg.norm(best_points, axis=1)
    if np.any(norms < 1e-10):
        best_points = seeds[0]
    
    return best_points
