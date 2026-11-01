import numpy as np
import scipy.spatial
import time
from scipy.optimize import minimize

N = 50

def _compute_facet_data(points):
    """
    Compute convex hull and facet circumcenters/radii (vectorized).
    Returns: hull, centers (array), radii (array), worst_idx
    """
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    if np.any(norms < 1e-15):
        return None, None, None, None
    points = points / norms
    
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
    # Normal to the facet
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
        # The normal from cross(a-b, b-c) might point inward or outward.
        # We check dot product with a. If negative, flip.
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
    
    # Vectorize gradient accumulation where possible, but loop over facets is fine if F is small (~100)
    # We can vectorize the contribution per facet if we accumulate into a sparse-like structure, 
    # but simple loop is safer and F is small.
    
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
        
        # Gradient of arccos(dot(C, p)) w.r.t. p is -C/sin(r)
        # Projected to tangent space at p: grad_t = grad - (grad . p) p
        # grad_contrib = -C / sin_r
        # We need to add w * grad_t to grads[p_idx] for p_idx in facet
        
        simplex = hull.simplices[i]
        # p_idx are 3 indices
        p_indices = simplex
        
        # Fetch points for these indices
        p_pts = points[p_indices]
        
        # grad_contrib is a vector (3,)
        grad_contrib = -C / sin_r
        
        # For each point in the facet, project grad_contrib to tangent space
        # radial_comp = (grad_contrib . p) * p
        # tangential = grad_contrib - radial_comp
        
        dots = np.sum(p_pts * grad_contrib, axis=1) # (3,)
        radial_comps = dots[:, np.newaxis] * p_pts   # (3, 3)
        tangential_grads = grad_contrib - radial_comps # (3, 3)
        
        # Add weighted tangential grads to the global gradient array
        # grads[p_indices] += w * tangential_grads
        for k in range(3):
            grads[p_indices[k]] += w * tangential_grads[k]
    
    return soft_radius, grads

def _project_to_sphere(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _lbfgs_stage(points, time_budget, beta_schedule):
    """
    Use L-BFGS-B to minimize the soft-max covering radius.
    Runs L-BFGS with a schedule of beta values (increasing sharpness).
    """
    start_time = time.time()
    current_pts = points.copy()
    current_radius = _compute_covering_radius(current_pts)
    best_pts = current_pts.copy()
    best_radius = current_radius
    
    for beta in beta_schedule:
        # Check time
        if time.time() - start_time > time_budget:
            break
            
        # Define objective and gradient for this beta
        def objective_and_grad(x, b=beta):
            if time.time() - start_time > time_budget:
                raise Exception("timeout")
            pts = _project_to_sphere(x.reshape(N, 3))
            soft_r, grads = _soft_max_radius_and_grad(pts, beta=b)
            return soft_r, grads.reshape(-1)
        
        try:
            # Use current best as start for this beta stage
            x0 = current_pts.reshape(-1)
            
            # Limit iterations to keep time under control
            # If we have lots of time, allow more iterations
            elapsed = time.time() - start_time
            remaining = time_budget - elapsed
            max_iter = int(100 + 100 * (remaining / time_budget)) # 100 to 200 iterations
            
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': max_iter, 'maxfun': max_iter*2, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
            )
            
            if result.success or result.fun < _compute_covering_radius(current_pts):
                new_pts = _project_to_sphere(result.x.reshape(N, 3))
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
    Directly minimizes the soft-max covering radius with high beta (approx max).
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
        new_points = _project_to_sphere(new_points)
        
        new_radius = _compute_covering_radius(new_points)
        
        if new_radius < true_radius - 1e-15:
            current_points = new_points
            step_size = min(step_size * growth, 0.1)
        else:
            step_size *= decay
            if step_size < min_step:
                break
    
    return best_points, best_radius

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
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        seeds.append(pts)
        
    # 3. Greedy initialization
    def greedy_init(seed_val):
        rng = np.random.default_rng(seed_val)
        pts = [rng.normal(size=3)]
        pts[0] = pts[0] / np.linalg.norm(pts[0])
        for _ in range(N - 1):
            candidates = rng.normal(size=(2000, 3))
            candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
            dots = candidates @ np.array(pts).T
            min_dots = np.max(dots, axis=1)
            idx = np.argmin(min_dots)
            pts.append(candidates[idx])
        return np.array(pts)
        
    seeds.append(greedy_init(101))
    seeds.append(greedy_init(202))
    seeds.append(greedy_init(303))
    
    return seeds

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_points = None
    best_radius = np.inf
    
    seeds = _generate_seeds()
    n_seeds = len(seeds)
    
    # Beta schedule for L-BFGS: Start smooth, get sharper
    # Low beta: global structure, High beta: local refinement
    beta_schedule = [20.0, 100.0, 500.0]
    
    # Allocate time per seed
    # We want to spend most time on the best few seeds
    # Strategy: Run all seeds with a quick pass, then refine the top 3 with more time
    
    # Pass 1: Quick optimization for all seeds
    time_per_seed_pass1 = 10.0
    
    results = []
    
    for i, seed_pts in enumerate(seeds):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        if remaining < 30: # Reserve 30s for final polish
            break
            
        # Budget for this seed in pass 1
        # If we have many seeds, limit time
        budget = min(time_per_seed_pass1, remaining / (n_seeds - i + 1))
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
    
    # Pass 2: Refine the top 3 candidates with more time and higher beta
    top_candidates = results[:3]
    
    # Reserve time for final polish of the absolute best
    # Let's say we spend 20s on final polish of the single best.
    # Remaining time for Pass 2: deadline - 20 - elapsed
    
    elapsed = time.time() - start_time
    remaining_for_pass2 = deadline - elapsed - 20.0
    
    if remaining_for_pass2 > 10:
        time_per_seed_pass2 = remaining_for_pass2 / len(top_candidates)
        
        for rad, pts, idx in top_candidates:
            if time.time() > deadline - 20:
                break
                
            # Use a sharper beta schedule for refinement
            # Start from where we left off or restart with high beta?
            # Restarting with high beta might get stuck. 
            # Better to continue from current best with increasing beta.
            # But _lbfgs_stage resets. 
            # Let's just run a new L-BFGS with high beta schedule.
            refine_schedule = [100.0, 500.0, 2000.0]
            
            refined_pts = _lbfgs_stage(pts, time_per_seed_pass2, refine_schedule)
            refined_rad = _compute_covering_radius(refined_pts)
            
            if refined_rad < best_radius:
                best_radius = refined_rad
                best_points = refined_pts.copy()
                
    # Pass 3: Final local polish on the single best
    if best_points is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        if remaining > 5:
            budget = min(18.0, remaining - 2)
            if budget > 1:
                polished_pts, polished_rad = _local_polish(best_points, budget, beta=2000.0)
                if polished_rad < best_radius:
                    best_radius = polished_rad
                    best_points = polished_pts.copy()
                    
    if best_points is None:
        best_points = seeds[0]
        
    # Ensure output is normalized
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points
