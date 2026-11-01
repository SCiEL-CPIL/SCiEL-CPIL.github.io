import numpy as np
import scipy.spatial
import time
from scipy.optimize import minimize

N = 50
TIME_LIMIT = 145.0  # safety margin before 150s kill

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _compute_facet_data(points):
    """
    Compute convex hull and facet circumcenters/radii (vectorized).
    Returns: hull, centers (array), radii (array), worst_idx
    """
    points = _normalize(points)
    
    try:
        hull = scipy.spatial.ConvexHull(points)
    except Exception:
        return None, None, None, None
    
    facets = hull.simplices
    n_facets = len(facets)
    if n_facets == 0:
        return hull, np.zeros((0,3)), np.zeros(0), None
        
    a = points[facets[:, 0]]  # (F, 3)
    b = points[facets[:, 1]]
    c = points[facets[:, 2]]
    
    # Vectorized circumcenter computation
    v1 = a - b
    v2 = b - c
    n = np.cross(v1, v2)
    norm_n = np.linalg.norm(n, axis=1)
    
    # Handle degenerate facets (collinear points)
    valid = norm_n > 1e-12
    
    centers = np.zeros((n_facets, 3))
    radii = np.zeros(n_facets)
    
    if np.any(valid):
        cc = n[valid] / norm_n[valid, np.newaxis]
        
        # Ensure circumcenter faces outward (toward 'a')
        dots = np.sum(cc * a[valid], axis=1)
        cc[dots < 0] *= -1
        
        centers[valid] = cc
        
        # Calculate radii (angular distance from center to vertex a)
        cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
        radii[valid] = np.arccos(cos_r)
    
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
    # Use log-sum-exp trick for numerical stability
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
        p_indices = simplex
        p_pts = points[p_indices]
        
        # Gradient of arccos(dot(C, p)) w.r.t. p is -C/sin(r)
        grad_contrib = -C / sin_r
        
        # Project to tangent space at each point in the facet
        dots = np.sum(p_pts * grad_contrib, axis=1)
        radial_comps = dots[:, np.newaxis] * p_pts
        tangential_grads = grad_contrib - radial_comps
        
        for k in range(3):
            grads[p_indices[k]] += w * tangential_grads[k]
    
    return soft_radius, grads

def _logdet_gram_and_grad(points):
    """
    Compute log-determinant of the Gram matrix and its gradient.
    The Gram matrix G = P @ P^T where P is (N, 3).
    Maximizing log(det(G)) encourages points to be spread out.
    """
    G = points @ points.T
    # Add small regularization for numerical stability
    G += 1e-12 * np.eye(N)
    
    try:
        sign, logdet = np.linalg.slogdet(G)
    except np.linalg.LinAlgError:
        return -np.inf, np.zeros_like(points)
    
    if sign <= 0:
        return -np.inf, np.zeros_like(points)
    
    # Gradient of log(det(G)) w.r.t. points
    # d log(det(G)) / d P = 2 * G^{-1} @ P
    try:
        G_inv = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        return logdet, np.zeros_like(points)
    
    # Gradient w.r.t. each row of P
    # d log(det(G)) / d P_i = 2 * (G^{-1} @ P)_i
    grad = 2.0 * (G_inv @ points)
    
    # Project gradient to tangent space
    # For each point p_i, the tangent space is orthogonal to p_i
    # grad_tangent_i = grad_i - (grad_i . p_i) * p_i
    dots = np.sum(grad * points, axis=1)
    radial_comps = dots[:, np.newaxis] * points
    grad_tangent = grad - radial_comps
    
    return logdet, grad_tangent

def _spectral_objective_and_grad(points, time_budget, alpha=1.0, beta=100.0):
    """
    Combined objective: soft_max_radius - alpha * logdet_gram
    This encourages both small covering radius and uniform distribution.
    """
    soft_r, grad_r = _soft_max_radius_and_grad(points, beta=beta)
    logdet, grad_logdet = _logdet_gram_and_grad(points)
    
    obj = soft_r - alpha * logdet
    grad = grad_r - alpha * grad_logdet
    
    return obj, grad

def _spectral_lbfgs(points, time_budget, alpha_schedule, beta_schedule):
    """
    L-BFGS optimization with combined spectral-geometric objective.
    """
    start_time = time.time()
    current_pts = points.copy()
    current_radius = _compute_covering_radius(current_pts)
    best_pts = current_pts.copy()
    best_radius = current_radius
    
    # Pair alpha and beta schedules
    if len(alpha_schedule) != len(beta_schedule):
        n = max(len(alpha_schedule), len(beta_schedule))
        while len(alpha_schedule) < n:
            alpha_schedule.append(alpha_schedule[-1])
        while len(beta_schedule) < n:
            beta_schedule.append(beta_schedule[-1])
    
    for alpha, beta in zip(alpha_schedule, beta_schedule):
        if time.time() - start_time > time_budget:
            break
            
        def objective_and_grad(x, a=alpha, b=beta):
            if time.time() - start_time > time_budget:
                raise Exception("timeout")
            pts = _normalize(x.reshape(N, 3))
            obj, grads = _spectral_objective_and_grad(pts, time_budget, alpha=a, beta=b)
            return obj, grads.reshape(-1)
        
        try:
            x0 = current_pts.reshape(-1)
            
            elapsed = time.time() - start_time
            remaining = time_budget - elapsed
            max_iter = int(80 + 120 * (remaining / time_budget))
            
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': max_iter, 'maxfun': max_iter*2, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
            )
            
            new_pts = _normalize(result.x.reshape(N, 3))
            new_rad = _compute_covering_radius(new_pts)
            
            if new_rad < current_radius:
                current_pts = new_pts
                current_radius = new_rad
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = current_pts.copy()
                
        except Exception:
            pass
            
    return best_pts

def _lbfgs_stage(points, time_budget, beta_schedule):
    """
    Use L-BFGS-B to minimize the soft-max covering radius.
    """
    start_time = time.time()
    current_pts = points.copy()
    current_radius = _compute_covering_radius(current_pts)
    best_pts = current_pts.copy()
    best_radius = current_radius
    
    for beta in beta_schedule:
        if time.time() - start_time > time_budget:
            break
            
        def objective_and_grad(x, b=beta):
            if time.time() - start_time > time_budget:
                raise Exception("timeout")
            pts = _normalize(x.reshape(N, 3))
            soft_r, grads = _soft_max_radius_and_grad(pts, beta=b)
            return soft_r, grads.reshape(-1)
        
        try:
            x0 = current_pts.reshape(-1)
            
            elapsed = time.time() - start_time
            remaining = time_budget - elapsed
            max_iter = int(50 + 100 * (remaining / time_budget))
            
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': max_iter, 'maxfun': max_iter*2, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
            )
            
            if result.success or result.fun < _compute_covering_radius(current_pts):
                new_pts = _normalize(result.x.reshape(N, 3))
                new_rad = _compute_covering_radius(new_pts)
                
                if new_rad < current_radius:
                    current_pts = new_pts
                    current_radius = new_rad
                
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = current_pts.copy()
                    
        except Exception:
            pass
            
    return best_pts

def _local_polish(points, time_budget, beta=1000.0):
    """
    Final local polish using projected gradient descent on tangent space.
    """
    current_points = points.copy()
    best_radius = _compute_covering_radius(current_points)
    best_points = current_points.copy()
    
    start_time = time.time()
    step_size = 0.02
    min_step = 1e-12
    decay = 0.85
    growth = 1.05
    
    while time.time() - start_time < time_budget:
        soft_radius, grads = _soft_max_radius_and_grad(current_points, beta=beta)
        
        true_radius = _compute_covering_radius(current_points)
        if true_radius < best_radius - 1e-14:
            best_radius = true_radius
            best_points = current_points.copy()
        
        grad_norm = np.linalg.norm(grads)
        if grad_norm < 1e-15:
            break
        
        direction = grads / grad_norm
        new_points = current_points + step_size * direction
        new_points = _normalize(new_points)
        
        new_radius = _compute_covering_radius(new_points)
        
        if new_radius < true_radius - 1e-15:
            current_points = new_points
            step_size = min(step_size * growth, 0.1)
        else:
            step_size *= decay
            if step_size < min_step:
                break
    
    return best_points, best_radius

def _worst_facet_optimization(points, time_budget):
    """
    Directly optimize the worst facet by moving its vertices away from the circumcenter.
    """
    start_time = time.time()
    current_pts = points.copy()
    best_radius = _compute_covering_radius(current_pts)
    best_pts = current_pts.copy()
    
    step_size = 0.05
    min_step = 1e-10
    decay = 0.9
    
    while time.time() - start_time < time_budget:
        hull, centers, radii, worst_idx = _compute_facet_data(current_pts)
        if hull is None or worst_idx is None:
            break
        
        r = radii[worst_idx]
        if r < 1e-12:
            break
        
        C = centers[worst_idx]
        sin_r = np.sin(r)
        if sin_r < 1e-10:
            break
        
        simplex = hull.simplices[worst_idx]
        p_indices = simplex
        p_pts = current_pts[p_indices]
        
        # Move each vertex away from the circumcenter direction
        grad_contrib = -C / sin_r
        
        dots = np.sum(p_pts * grad_contrib, axis=1)
        radial_comps = dots[:, np.newaxis] * p_pts
        tangential_grads = grad_contrib - radial_comps
        
        # Apply step
        new_pts = current_pts.copy()
        for k in range(3):
            new_pts[p_indices[k]] = current_pts[p_indices[k]] + step_size * tangential_grads[k]
        new_pts = _normalize(new_pts)
        
        new_radius = _compute_covering_radius(new_pts)
        
        if new_radius < best_radius - 1e-14:
            best_radius = new_radius
            best_pts = new_pts.copy()
            current_pts = new_pts
            step_size = min(step_size * 1.05, 0.1)
        else:
            step_size *= decay
            if step_size < min_step:
                break
    
    return best_pts

def _generate_seeds():
    seeds = []
    
    # 1. Fibonacci sphere
    j = np.arange(N) + 0.5
    z = 1.0 - 2.0 * j / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * j
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    seeds.append(fib)
    
    # 2. Random starts
    for seed_val in [42, 1, 2, 3, 7, 13, 99, 123, 555, 777]:
        rng = np.random.default_rng(seed_val)
        pts = rng.normal(size=(N, 3))
        pts = _normalize(pts)
        seeds.append(pts)
        
    # 3. Greedy initialization
    def greedy_init(seed_val):
        rng = np.random.default_rng(seed_val)
        pts = [rng.normal(size=3)]
        pts[0] = pts[0] / np.linalg.norm(pts[0])
        for _ in range(N - 1):
            candidates = rng.normal(size=(2000, 3))
            candidates = _normalize(candidates)
            dots = candidates @ np.array(pts).T
            min_dots = np.max(dots, axis=1)
            idx = np.argmin(min_dots)
            pts.append(candidates[idx])
        return np.array(pts)
        
    seeds.append(greedy_init(101))
    seeds.append(greedy_init(202))
    seeds.append(greedy_init(303))
    
    # 4. Start from icosahedron-like structure and subdivide
    phi_golden = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1]
    ], dtype=float)
    ico = _normalize(ico)
    
    # Pad icosahedron to 50 points by adding points near each vertex
    ico_extended = [ico[i] for i in range(12)]
    rng = np.random.default_rng(42)
    while len(ico_extended) < N:
        base = ico_extended[len(ico_extended) % 12]
        perturbation = rng.normal(scale=0.1, size=3)
        new_pt = base + perturbation
        ico_extended.append(new_pt / np.linalg.norm(new_pt))
    seeds.append(np.array(ico_extended))
    
    return seeds

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_points = None
    best_radius = np.inf
    
    seeds = _generate_seeds()
    n_seeds = len(seeds)
    
    # Strategy:
    # 1. Quick pass with L-BFGS on all seeds
    # 2. Spectral-geometric L-BFGS on top candidates
    # 3. Worst-facet optimization on top candidates
    # 4. Final local polish on the single best
    
    # Pass 1: Quick L-BFGS optimization for all seeds
    beta_schedule = [20.0, 100.0, 500.0]
    results = []
    
    for i, seed_pts in enumerate(seeds):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        if remaining < 30:  # Reserve 30s for later stages
            break
        
        budget = min(6.0, remaining / (n_seeds - i + 1))
        if budget < 2.0:
            break
        
        pts = _lbfgs_stage(seed_pts, budget, beta_schedule)
        rad = _compute_covering_radius(pts)
        results.append((rad, pts, i))
        
        if rad < best_radius:
            best_radius = rad
            best_points = pts.copy()
            
    # Sort results by radius
    results.sort(key=lambda x: x[0])
    
    # Pass 2: Spectral-geometric L-BFGS on top 5 candidates
    top_candidates = results[:5]
    
    elapsed = time.time() - start_time
    remaining_for_pass2 = deadline - elapsed - 20.0  # Reserve 20s for final stages
    
    if remaining_for_pass2 > 10:
        time_per_seed_pass2 = remaining_for_pass2 / len(top_candidates)
        
        for rad, pts, idx in top_candidates:
            if time.time() > deadline - 20:
                break
                
            # Spectral-geometric optimization
            alpha_schedule = [0.01, 0.05, 0.1, 0.5]
            beta_schedule_spec = [50.0, 100.0, 200.0, 500.0]
            spectral_pts = _spectral_lbfgs(pts, time_per_seed_pass2 * 0.6, alpha_schedule, beta_schedule_spec)
            spectral_rad = _compute_covering_radius(spectral_pts)
            
            if spectral_rad < best_radius:
                best_radius = spectral_rad
                best_points = spectral_pts.copy()
            
            # Worst facet optimization
            if time.time() < deadline - 20:
                worst_opt_pts = _worst_facet_optimization(spectral_pts, time_per_seed_pass2 * 0.4)
                worst_opt_rad = _compute_covering_radius(worst_opt_pts)
                
                if worst_opt_rad < best_radius:
                    best_radius = worst_opt_rad
                    best_points = worst_opt_pts.copy()
                
    # Pass 3: Final local polish on the single best
    if best_points is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        if remaining > 5:
            budget = min(15.0, remaining - 2)
            if budget > 1:
                # Local polish with high beta
                polished_pts, polished_rad = _local_polish(best_points, budget * 0.5, beta=2000.0)
                if polished_rad < best_radius:
                    best_radius = polished_rad
                    best_points = polished_pts.copy()
                
                # Worst facet optimization on the polished result
                if time.time() < deadline - 2:
                    worst_opt_pts = _worst_facet_optimization(best_points, budget * 0.5)
                    worst_opt_rad = _compute_covering_radius(worst_opt_pts)
                    
                    if worst_opt_rad < best_radius:
                        best_radius = worst_opt_rad
                        best_points = worst_opt_pts.copy()
                    
    if best_points is None:
        best_points = seeds[0]
        
    # Ensure output is normalized
    best_points = _normalize(best_points)
    
    return best_points
