import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50
TIME_LIMIT = 145.0  # safety margin from 150s

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _compute_facet_data(points):
    """
    Compute convex hull and facet circumcenters/radii.
    Returns: hull, centers (array), radii (array), worst_idx
    """
    points = _normalize(points)
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return None, None, None, None
    
    n_facets = len(hull.simplices)
    centers = np.zeros((n_facets, 3))
    radii = np.zeros(n_facets)
    
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
                continue
            centers[i] = c_dir / norm_c_dir
            radii[i] = np.arccos(np.clip(np.dot(centers[i], a), -1.0, 1.0))
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
    Uses a more numerically stable approach.
    """
    hull, centers, radii, worst_idx = _compute_facet_data(points)
    if hull is None or worst_idx is None:
        return np.pi, np.zeros_like(points)
    
    max_r = np.max(radii)
    shifted = beta * (radii - max_r)
    max_shifted = np.max(shifted)
    exp_terms = np.exp(shifted - max_shifted)
    log_sum = max_shifted + np.log(np.sum(exp_terms))
    soft_radius = max_r + log_sum / beta
    weights = exp_terms / np.sum(exp_terms)
    
    grads = np.zeros_like(points)
    n_facets = len(hull.simplices)
    
    for i in range(n_facets):
        w = weights[i]
        if w < 1e-15:
            continue
        r = radii[i]
        if r < 1e-12:
            continue
        
        C = centers[i]
        sin_r = np.sin(r)
        if sin_r < 1e-10:
            continue
        
        grad_contrib = -C / sin_r
        simplex = hull.simplices[i]
        for k in range(3):
            p_idx = simplex[k]
            p = points[p_idx]
            radial = np.dot(grad_contrib, p) * p
            tangential = grad_contrib - radial
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
    """
    start_time = time.time()
    pts = points.copy()
    best_pts = pts.copy()
    best_radius = _compute_covering_radius(pts)
    step = 0.005
    
    for _ in range(iterations):
        if time.time() - start_time > time_budget:
            break
        
        # Vectorized pairwise repulsion
        # diff[i,j] = pts[i] - pts[j]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero
        
        # Force: diff / dist^3 = diff / (dist_sq * sqrt(dist_sq))
        dist_cubed = dist_sq * np.sqrt(dist_sq)
        forces_raw = diff / dist_cubed[:, :, np.newaxis]
        
        # Project to tangent space: force - (force . pts[i]) * pts[i]
        radial_dots = np.sum(forces_raw * pts[:, np.newaxis, :], axis=2)
        forces_tangent = forces_raw - radial_dots[:, :, np.newaxis] * pts[:, np.newaxis, :]
        
        # Sum over all j for each i
        forces = np.sum(forces_tangent, axis=1)
        
        # Normalize forces
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms = np.maximum(force_norms, 1e-15)
        normalized_forces = forces / force_norms
        
        new_pts = pts + step * normalized_forces
        new_pts = _project_to_sphere(new_pts)
        
        new_radius = _compute_covering_radius(new_pts)
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            step *= 0.8
            if step < 1e-6:
                break
    
    return best_pts, best_radius

def _lbfgs_stage(points, time_budget):
    """
    Use L-BFGS-B to minimize the soft-max covering radius.
    Uses many beta stages for progressive sharpening.
    """
    start_time = time.time()
    best_pts = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Progressive beta stages from smooth to sharp
    betas = [20.0, 40.0, 80.0, 160.0, 320.0, 640.0]
    
    for beta in betas:
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 3.0:
            break
        
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
                options={'maxiter': 400, 'maxfun': 3000, 'ftol': 1e-14, 'gtol': 1e-11, 'maxls': 40}
            )
            new_pts = _project_to_sphere(result.x.reshape(N, 3))
            new_radius = _compute_covering_radius(new_pts)
            if new_radius < best_radius - 1e-12:
                best_radius = new_radius
                best_pts = new_pts
        except Exception:
            pass
    
    return best_pts

def _local_polish(points, time_budget, beta=100.0):
    """
    Final local polish using projected gradient descent on tangent space.
    Directly minimizes the soft-max covering radius.
    Uses adaptive step size with line search.
    """
    current_points = points.copy()
    best_radius = _compute_covering_radius(current_points)
    best_points = current_points.copy()
    
    start_time = time.time()
    step_size = 0.02
    min_step = 1e-14
    decay = 0.8
    growth = 1.1
    stagnant = 0
    
    while time.time() - start_time < time_budget:
        soft_radius, grads = _soft_max_radius_and_grad(current_points, beta=beta)
        
        true_radius = _compute_covering_radius(current_points)
        if true_radius < best_radius - 1e-14:
            best_radius = true_radius
            best_points = current_points.copy()
            stagnant = 0
        else:
            stagnant += 1
        
        if stagnant > 80:
            break
        
        grad_norm = np.linalg.norm(grads)
        if grad_norm < 1e-15:
            break
        
        direction = -grads / grad_norm
        
        # Backtracking line search
        alpha = step_size
        accepted = False
        for _ in range(10):
            new_points = current_points + alpha * direction
            new_points = _project_to_sphere(new_points)
            new_radius = _compute_covering_radius(new_points)
            
            if new_radius < true_radius - 1e-15:
                current_points = new_points
                step_size = min(alpha * 1.5, 0.15)
                accepted = True
                break
            alpha *= 0.5
        
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
    for seed_val in [42, 1, 2, 3, 7, 13, 99, 123, 555, 777, 2024, 31415]:
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
            pts, rad = _repulsive_relaxation(seed_pts, min(2.5, remaining - 2))
        else:
            pts = seed_pts
            rad = _compute_covering_radius(pts)
        
        # Step 2: L-BFGS optimization with progressive beta
        elapsed = time.time() - start_time
        remaining = stage1_deadline - elapsed
        if remaining > 3:
            pts = _lbfgs_stage(pts, min(5.0, remaining - 2))
        
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
        # Worst-facet optimization with adaptive step sizes
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 15:
            for step in [0.05, 0.02, 0.01, 0.005]:
                elapsed = time.time() - start_time
                remaining = deadline - elapsed
                if remaining < 5:
                    break
                pts = _worst_facet_optimization(best_points, step, min(4.0, remaining - 3), iterations=80)
                rad = _compute_covering_radius(pts)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Lloyd's iteration for local refinement
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 10:
            pts, rad = _lloyd_iteration(best_points, min(6.0, remaining - 5), iterations=40)
            if rad < best_radius:
                best_radius = rad
                best_points = pts.copy()
        
        # Fine local polish with moderate beta
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 10:
            budget = min(8.0, remaining - 5)
            if budget > 4:
                pts, rad = _local_polish(best_points, budget, beta=200.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Very fine polish with high beta for final convergence
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 8:
            budget = min(6.0, remaining - 4)
            if budget > 3:
                pts, rad = _local_polish(best_points, budget, beta=500.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Ultra-fine polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 5:
            budget = min(4.0, remaining - 3)
            if budget > 2:
                pts, rad = _local_polish(best_points, budget, beta=1000.0)
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
