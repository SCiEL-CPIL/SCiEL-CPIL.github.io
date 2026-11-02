import numpy as np
import scipy.spatial
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 50
TIME_LIMIT = 143.0  # safety margin from 150s

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
        
        simplex = hull.simplices[i]
        for k in range(3):
            p_idx = simplex[k]
            p = points[p_idx]
            grad_contrib = -C / sin_r
            radial = np.dot(grad_contrib, p) * p
            tangential = grad_contrib - radial
            grads[p_idx] += w * tangential
    
    return soft_radius, grads

def _project_to_sphere(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _facet_targeted_move(points, time_budget, max_iters=2000):
    """
    Directly move vertices of worst facets toward their circumcenters.
    This is a more direct approach than gradient descent for the worst facet.
    """
    current = points.copy()
    best_radius = _compute_covering_radius(current)
    best_points = current.copy()
    
    start_time = time.time()
    step_size = 0.05
    min_step = 1e-12
    decay = 0.8
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        hull, centers, radii, worst_idx = _compute_facet_data(current)
        if hull is None or worst_idx is None:
            break
        
        current_radius = radii[worst_idx]
        if current_radius < best_radius - 1e-14:
            best_radius = current_radius
            best_points = current.copy()
        
        # Get the worst facet
        simplex = hull.simplices[worst_idx]
        C = centers[worst_idx]
        r = radii[worst_idx]
        
        if r < 1e-10:
            break
        
        # Move each vertex of the worst facet toward the circumcenter
        improved = False
        for k in range(3):
            p_idx = simplex[k]
            p = current[p_idx]
            
            # Direction toward circumcenter
            d = C - p
            d_norm = np.linalg.norm(d)
            if d_norm < 1e-12:
                continue
            d = d / d_norm
            
            # Try multiple step sizes
            best_new_radius = current_radius
            best_new_p = p
            
            for scale in [step_size, step_size * 0.5, step_size * 0.25]:
                new_p = p + scale * d
                new_p = new_p / np.linalg.norm(new_p)
                
                test_current = current.copy()
                test_current[p_idx] = new_p
                
                new_radius = _compute_covering_radius(test_current)
                if new_radius < best_new_radius - 1e-14:
                    best_new_radius = new_radius
                    best_new_p = new_p
            
            if best_new_p is not p:
                current[p_idx] = best_new_p
                improved = True
                step_size = min(step_size * 1.1, 0.08)
            else:
                step_size *= decay
        
        if not improved:
            step_size *= decay
            if step_size < min_step:
                break
    
    return best_points, best_radius

def _lloyd_iteration(points, time_budget, max_iters=500):
    """
    Lloyd's algorithm: for each point, move it toward the centroid of its Voronoi cell
    (intersection of caps centered at other points).
    """
    current = points.copy()
    best_radius = _compute_covering_radius(current)
    best_points = current.copy()
    
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        # For each point, find the "center of mass" of its Voronoi cell
        # Approximate by finding the point on the sphere that maximizes the minimum
        # angular distance to all other points, weighted by proximity to the current point
        for i in range(N):
            p = current[i]
            
            # Find the farthest point from p among all other points
            dots = current @ p
            # Exclude self
            dots[i] = -2.0
            nearest_j = np.argmin(dots)  # farthest in angular distance
            
            # The Voronoi centroid should be somewhere between p and the direction
            # that maximizes the minimum distance to all other points
            # Approximate: move p slightly away from its nearest neighbor
            nn = current[nearest_j]
            away = p - nn
            away_norm = np.linalg.norm(away)
            if away_norm > 1e-12:
                away = away / away_norm
                new_p = p + 0.1 * away
                new_p = new_p / np.linalg.norm(new_p)
                
                test_current = current.copy()
                test_current[i] = new_p
                new_radius = _compute_covering_radius(test_current)
                
                if new_radius < best_radius - 1e-14:
                    current[i] = new_p
                    if new_radius < best_radius:
                        best_radius = new_radius
                        best_points = current.copy()
    
    return best_points, best_radius

def _repulsion_relax(points, time_budget, max_iters=3000):
    """
    Simple repulsion-based relaxation: each pair of points repels each other,
    and each point is attracted toward the center of its Voronoi cell.
    """
    current = points.copy()
    best_radius = _compute_covering_radius(current)
    best_points = current.copy()
    
    start_time = time.time()
    step_size = 0.01
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        forces = np.zeros_like(current)
        
        for i in range(N):
            p_i = current[i]
            
            # Find the point's "empty space" - direction where it can move
            # to increase the minimum distance to others
            dots = current @ p_i
            dots[i] = -2.0
            nn_j = np.argmin(dots)
            
            # Move away from nearest neighbor
            nn = current[nn_j]
            away = p_i - nn
            away_norm = np.linalg.norm(away)
            if away_norm > 1e-12:
                away = away / away_norm
                forces[i] += 0.5 * away
            
            # Also consider second nearest
            dots2 = dots.copy()
            dots2[nn_j] = -2.0
            nn2_j = np.argmin(dots2)
            nn2 = current[nn2_j]
            away2 = p_i - nn2
            away2_norm = np.linalg.norm(away2)
            if away2_norm > 1e-12:
                away2 = away2 / away2_norm
                forces[i] += 0.3 * away2
        
        # Update
        new_current = current + step_size * forces
        new_current = _project_to_sphere(new_current)
        
        new_radius = _compute_covering_radius(new_current)
        if new_radius < best_radius - 1e-14:
            current = new_current
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = current.copy()
        else:
            step_size *= 0.9
    
    return best_points, best_radius

def _lbfgs_stage(points, time_budget):
    """
    Use L-BFGS-B to minimize the soft-max covering radius.
    """
    start_time = time.time()
    best_pts = points.copy()
    best_soft_r = _compute_covering_radius(points)
    
    for beta in [80.0, 150.0, 300.0]:
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 3.0:
            break
        
        def objective_and_grad(x):
            if time.time() - start_time > time_budget:
                raise Exception("timeout")
            pts = _project_to_sphere(x.reshape(N, 3))
            soft_r, grads = _soft_max_radius_and_grad(pts, beta=beta)
            grad_flat = grads.reshape(-1)
            return soft_r, grad_flat
        
        x0 = best_pts.reshape(-1)
        
        try:
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 600, 'maxfun': 4000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 40}
            )
            if result.fun < best_soft_r - 1e-12:
                best_pts = _project_to_sphere(result.x.reshape(N, 3))
                best_soft_r = _compute_covering_radius(best_pts)
        except Exception:
            pass
    
    return best_pts

def _local_polish(points, time_budget, beta=100.0):
    """
    Final local polish using projected gradient descent on tangent space.
    """
    current_points = points.copy()
    best_radius = _compute_covering_radius(current_points)
    best_points = current_points.copy()
    
    start_time = time.time()
    step_size = 0.02
    min_step = 1e-14
    decay = 0.7
    growth = 1.15
    stagnant = 0
    max_stagnant = 150
    momentum = 0.4
    velocity = np.zeros_like(points)
    
    while time.time() - start_time < time_budget:
        soft_radius, grads = _soft_max_radius_and_grad(current_points, beta=beta)
        
        true_radius = _compute_covering_radius(current_points)
        if true_radius < best_radius - 1e-14:
            best_radius = true_radius
            best_points = current_points.copy()
            stagnant = 0
        else:
            stagnant += 1
        
        if stagnant > max_stagnant:
            break
        
        grad_norm = np.linalg.norm(grads)
        if grad_norm < 1e-16:
            break
        
        velocity = momentum * velocity - (1 - momentum) * grads / grad_norm
        
        # Try multiple step sizes
        best_new_radius = true_radius
        best_new_points = current_points
        
        for trial_scale in [step_size, step_size * 0.5, step_size * 0.25]:
            new_points = current_points + trial_scale * velocity
            new_points = _project_to_sphere(new_points)
            new_radius = _compute_covering_radius(new_points)
            if new_radius < best_new_radius - 1e-16:
                best_new_radius = new_radius
                best_new_points = new_points
        
        if best_new_points is not current_points:
            current_points = best_new_points
            step_size = min(step_size * growth, 0.06)
        else:
            step_size *= decay
            velocity *= 0.5
            if step_size < min_step:
                break
    
    return best_points, best_radius

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
    
    # Stage 1: Multi-stage optimization for each seed
    stage1_deadline = deadline - 35.0
    
    for i, seed_pts in enumerate(seeds):
        elapsed = time.time() - start_time
        remaining = stage1_deadline - elapsed
        
        if remaining < 5:
            break
        
        if i < n_seeds - 1:
            time_budget = min(5.0, remaining / (n_seeds - i))
        else:
            time_budget = min(6.0, remaining)
        
        if time_budget < 3.0:
            break
        
        # Apply L-BFGS stage
        pts = _lbfgs_stage(seed_pts, time_budget)
        
        # Facet-targeted move
        elapsed = time.time() - start_time
        remaining = stage1_deadline - elapsed
        if remaining > 3:
            pts, rad = _facet_targeted_move(pts, min(2.0, remaining - 2))
        else:
            rad = _compute_covering_radius(pts)
        
        if rad < best_radius:
            best_radius = rad
            best_points = pts.copy()
    
    # Stage 2: Additional optimization on best candidates
    if best_points is not None:
        # Try repulsion relax on best
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 25:
            budget = min(10.0, remaining - 20)
            if budget > 4:
                pts, rad = _repulsion_relax(best_points, budget)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Lloyd iteration
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 20:
            budget = min(8.0, remaining - 15)
            if budget > 3:
                pts, rad = _lloyd_iteration(best_points, budget)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Fine local polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 15:
            budget = min(10.0, remaining - 12)
            if budget > 4:
                pts, rad = _local_polish(best_points, budget, beta=200.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Very fine polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 10:
            budget = min(7.0, remaining - 8)
            if budget > 3:
                pts, rad = _local_polish(best_points, budget, beta=500.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Ultra-fine polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 7:
            budget = min(5.0, remaining - 5)
            if budget > 2:
                pts, rad = _local_polish(best_points, budget, beta=1000.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Final high-precision polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 4:
            budget = min(3.0, remaining - 3)
            if budget > 1.5:
                pts, rad = _local_polish(best_points, budget, beta=2000.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
    
    if best_points is None:
        best_points = seeds[0]
    
    # Ensure output is normalized and valid
    best_points = _normalize(best_points)
    
    # Final sanity check
    norms = np.linalg.norm(best_points, axis=1)
    if np.any(norms < 1e-10):
        best_points = seeds[0]
    
    return best_points
