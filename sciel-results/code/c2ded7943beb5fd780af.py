import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _compute_facet_data(points):
    """
    Compute convex hull and facet circumcenters/radii.
    Returns: hull, centers (array), radii (array)
    """
    points = _normalize(points)
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return None, None, None
    
    n_facets = len(hull.simplices)
    centers = np.zeros((n_facets, 3))
    radii = np.zeros(n_facets)
    
    for i, simplex in enumerate(hull.simplices):
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        
        # Use Gram matrix approach for stability
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
                # Degenerate, use centroid direction
                c_dir = (a + b + c)
                norm_c_dir = np.linalg.norm(c_dir)
                if norm_c_dir < 1e-10:
                    centers[i] = a
                    radii[i] = 0
                    continue
                centers[i] = c_dir / norm_c_dir
                radii[i] = np.arccos(np.clip(np.dot(centers[i], a), -1.0, 1.0))
                continue
            centers[i] = c_dir / norm_c_dir
            radii[i] = np.arccos(np.clip(np.dot(centers[i], a), -1.0, 1.0))
        except np.linalg.LinAlgError:
            # Fallback to cross product
            n = np.cross(b - a, c - b)
            norm = np.linalg.norm(n)
            if norm < 1e-10:
                centers[i] = a
                radii[i] = 0
            else:
                centers[i] = n / norm
                radii[i] = np.arccos(np.clip(np.dot(centers[i], a), -1.0, 1.0))
    
    return hull, centers, radii

def _covering_radius(points):
    """Compute the covering radius (max facet circumradius)."""
    hull, centers, radii = _compute_facet_data(points)
    if hull is None or len(radii) == 0:
        return np.pi
    return np.max(radii)

def _soft_max_radius_and_grad(points, beta=50.0):
    """
    Compute soft-max radius and its gradient w.r.t. points.
    Uses log-sum-exp for numerical stability.
    """
    hull, centers, radii = _compute_facet_data(points)
    if hull is None or len(radii) == 0:
        return np.pi, np.zeros_like(points)
    
    max_r = np.max(radii)
    
    # Log-sum-exp trick
    shifted = beta * (radii - max_r)
    exp_terms = np.exp(shifted - np.max(shifted))
    log_sum = np.max(shifted) + np.log(np.sum(exp_terms))
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
        
        # Gradient of arccos(dot(C, p)) w.r.t. p is -C/sin(r) projected to tangent space
        simplex = hull.simplices[i]
        for k in range(3):
            p_idx = simplex[k]
            p = points[p_idx]
            grad_contrib = -C / sin_r
            # Project to tangent space at p
            radial = np.dot(grad_contrib, p) * p
            tangential = grad_contrib - radial
            grads[p_idx] += w * tangential
    
    return soft_radius, grads

def _repulsion_relax(pts, time_budget, alpha_start=0.05):
    """
    Repulsion relaxation: minimize sum 1/||p_i - p_j||^2 via gradient descent.
    Helps spread points more uniformly before covering optimization.
    """
    pts = pts.copy()
    alpha = alpha_start
    start_t = time.time()
    
    def repulsion_energy(p):
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        mask = ~np.eye(N, dtype=bool)
        inv_dist_sq = np.zeros_like(dist_sq)
        np.divide(1.0, dist_sq, out=inv_dist_sq, where=mask)
        return 0.5 * np.sum(inv_dist_sq)
    
    current_energy = repulsion_energy(pts)
    
    while time.time() - start_t < time_budget:
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        
        mask = np.eye(N, dtype=bool)
        inv_dist_sq = np.zeros_like(dist_sq)
        np.divide(1.0, dist_sq, out=inv_dist_sq, where=~mask)
        
        forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        # Project onto tangent plane
        dot_fp = np.sum(forces * pts, axis=1)
        forces_tan = forces - dot_fp[:, np.newaxis] * pts
        
        # Adaptive step with backtracking
        step = alpha
        improved = False
        for _ in range(8):
            new_pts = pts + step * forces_tan
            new_pts = _normalize(new_pts)
            new_energy = repulsion_energy(new_pts)
            if new_energy < current_energy:
                pts = new_pts
                current_energy = new_energy
                improved = True
                break
            step *= 0.5
        
        if not improved:
            alpha *= 0.85
        else:
            alpha = min(alpha * 1.02, 0.08)
        
        alpha = max(alpha, 1e-6)
    
    return pts

def _lloyd_iteration(pts, time_budget, step_size=0.03):
    """
    Weighted Lloyd iteration: move each point toward weighted average of
    circumcenters of facets it belongs to, weighted by facet radius.
    """
    pts = pts.copy()
    start_t = time.time()
    
    while time.time() - start_t < time_budget:
        hull, centers, radii = _compute_facet_data(pts)
        if hull is None:
            break
        
        max_rad = np.max(radii)
        if max_rad < 1e-10:
            break
        
        updates = np.zeros_like(pts)
        weights_sum = np.zeros(N)
        
        for f in range(len(hull.simplices)):
            rad = radii[f]
            if rad < 1e-12:
                continue
            # Weight: higher for larger facets
            w = (rad / max_rad) ** 2
            cc = centers[f]
            for idx in hull.simplices[f]:
                updates[idx] += w * (cc - pts[idx])
                weights_sum[idx] += w
        
        mask = weights_sum > 1e-10
        if np.any(mask):
            pts[mask] += step_size * updates[mask] / weights_sum[mask, np.newaxis]
            pts = _normalize(pts)
    
    return pts

def _lbfgs_stage(points, time_budget):
    """
    Use L-BFGS-B to minimize the soft-max covering radius.
    Uses adaptive beta: starts moderate, increases for finer convergence.
    """
    start_time = time.time()
    best_pts = points.copy()
    best_soft_r = _covering_radius(points)
    
    for beta in [80.0, 150.0, 300.0]:
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 3.0:
            break
        
        def objective_and_grad(x):
            if time.time() - start_time > time_budget:
                raise Exception("timeout")
            pts = _normalize(x.reshape(N, 3))
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
                options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.success or result.fun < best_soft_r:
                new_pts = _normalize(result.x.reshape(N, 3))
                new_radius = _covering_radius(new_pts)
                if new_radius < best_soft_r:
                    best_pts = new_pts
                    best_soft_r = new_radius
        except Exception:
            pass
    
    return best_pts, best_soft_r

def _optimize(points, time_budget):
    """
    Multi-stage optimization:
    1. Repulsion relaxation for uniform spread
    2. Lloyd iteration for covering-specific improvement
    3. L-BFGS with soft-max for precise local optimization
    """
    start_time = time.time()
    points = _normalize(points.copy())
    
    # Stage 1: Repulsion relaxation (short budget)
    rep_budget = min(5.0, time_budget * 0.15)
    if rep_budget > 1.0:
        points = _repulsion_relax(points, rep_budget)
    
    # Stage 2: Lloyd iteration (medium budget)
    lloyd_budget = min(10.0, time_budget * 0.3)
    elapsed = time.time() - start_time
    remaining = time_budget - elapsed
    if remaining > 2.0:
        points = _lloyd_iteration(points, min(lloyd_budget, remaining - 1.0))
    
    # Stage 3: L-BFGS with soft-max (remaining budget)
    elapsed = time.time() - start_time
    remaining = time_budget - elapsed
    if remaining > 2.0:
        points, radius = _lbfgs_stage(points, remaining - 0.5)
    else:
        radius = _covering_radius(points)
    
    return points, radius

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety
    
    best_points = None
    best_radius = np.pi
    
    # Generate multiple starting configurations
    starts = []
    
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    starts.append(init_fib)
    
    # 2. Random starts with different seeds
    rng = np.random.default_rng(42)
    for seed in range(8):
        if time.time() > deadline - 10:
            break
        rng = np.random.default_rng(42 + seed)
        p = rng.normal(size=(N, 3))
        p = _normalize(p)
        starts.append(p)
    
    # 3. Perturbed Fibonacci
    rng = np.random.default_rng(123)
    for _ in range(4):
        if time.time() > deadline - 10:
            break
        p = init_fib + rng.normal(scale=0.05, size=(N, 3))
        p = _normalize(p)
        starts.append(p)
    
    # Optimize each start
    for start_idx, start in enumerate(starts):
        if time.time() > deadline - 5:
            break
            
        remaining = deadline - time.time()
        if remaining < 5:
            break
            
        # Allocate more time to earlier starts (they tend to be better)
        if start_idx == 0:
            budget = min(remaining - 2, 40)
        elif start_idx < 4:
            budget = min(remaining - 2, 25)
        else:
            budget = min(remaining - 2, 15)
        
        opt_points, radius = _optimize(start, budget)
        
        if radius < best_radius:
            best_radius = radius
            best_points = opt_points
    
    if best_points is None:
        return init_fib
        
    return _normalize(best_points)
