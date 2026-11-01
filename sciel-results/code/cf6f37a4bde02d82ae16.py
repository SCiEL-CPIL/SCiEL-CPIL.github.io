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

def _electrostatic_energy_and_grad(points, power=4.0, penalty_weight=0.1):
    """
    Compute electrostatic repulsion energy and gradient.
    Energy = sum_{i<j} 1/d(i,j)^power + penalty_weight * sum of facet circumradius penalties
    """
    n = len(points)
    energy = 0.0
    grads = np.zeros_like(points)
    
    # Pairwise electrostatic repulsion
    for i in range(n):
        for j in range(i+1, n):
            diff = points[i] - points[j]
            d2 = np.dot(diff, diff)
            if d2 < 1e-15:
                d2 = 1e-15
            d = np.sqrt(d2)
            # Energy: 1/d^power
            e = 1.0 / (d ** power)
            energy += e
            # Gradient: dE/dd = -power/d^(power+1), dE/dp_i = -power/d^(power+1) * (p_i - p_j)/d
            # = -power/d^(power+2) * (p_i - p_j)
            grad_factor = -power / (d2 ** ((power+1)/2.0))
            grads[i] += grad_factor * diff
            grads[j] -= grad_factor * diff
    
    # Add penalty for large facet circumradii
    hull, centers, radii, worst_idx = _compute_facet_data(points)
    if hull is not None:
        # Penalty = penalty_weight * sum(radii^2) to push all facets to be small
        penalty = penalty_weight * np.sum(radii ** 2)
        energy += penalty
        
        # Gradient of penalty
        for i in range(len(hull.simplices)):
            r = radii[i]
            if r < 1e-12:
                continue
            C = centers[i]
            sin_r = np.sin(r)
            if sin_r < 1e-10:
                continue
            
            # d(penalty)/d(points) = penalty_weight * 2 * r * d(r)/d(points)
            # d(r)/d(p) = -C/sin(r) projected to tangent
            simplex = hull.simplices[i]
            p_pts = points[simplex]
            
            grad_contrib = -2.0 * penalty_weight * r * C / sin_r
            
            dots = np.sum(p_pts * grad_contrib, axis=1)
            radial_comps = dots[:, np.newaxis] * p_pts
            tangential_grads = grad_contrib - radial_comps
            
            for k in range(3):
                grads[simplex[k]] += tangential_grads[k]
    
    return energy, grads

def _electrostatic_relax(points, time_budget, power=4.0, penalty_weight=0.1, step_size=0.01):
    """
    Physics-based electrostatic relaxation with gradient descent.
    """
    start_time = time.time()
    current_pts = points.copy()
    best_radius = _compute_covering_radius(current_pts)
    best_pts = current_pts.copy()
    
    step = step_size
    min_step = 1e-12
    decay = 0.9
    growth = 1.05
    
    while time.time() - start_time < time_budget:
        energy, grads = _electrostatic_energy_and_grad(current_pts, power=power, penalty_weight=penalty_weight)
        
        true_radius = _compute_covering_radius(current_pts)
        if true_radius < best_radius - 1e-14:
            best_radius = true_radius
            best_pts = current_pts.copy()
        
        grad_norm = np.linalg.norm(grads)
        if grad_norm < 1e-15:
            break
        
        direction = grads / grad_norm
        new_pts = current_pts - step * direction  # descent
        new_pts = _normalize(new_pts)
        
        new_radius = _compute_covering_radius(new_pts)
        
        if new_radius < true_radius - 1e-15:
            current_pts = new_pts
            step = min(step * growth, 0.05)
        else:
            step *= decay
            if step < min_step:
                break
    
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
        # Gradient direction for each vertex: -C/sin(r) projected to tangent
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

def _lloyd_iteration(points, time_budget, n_iter=50):
    """
    Lloyd's algorithm: iteratively move each point to the centroid of its Voronoi cell.
    On the sphere, approximate by moving toward the average of nearby points' antipodes.
    """
    start_time = time.time()
    current_pts = points.copy()
    best_radius = _compute_covering_radius(current_pts)
    best_pts = current_pts.copy()
    
    for _ in range(n_iter):
        if time.time() - start_time > time_budget:
            break
        
        new_pts = current_pts.copy()
        
        for i in range(N):
            # Find the "Voronoi region" approximately: points closer to i than to any other
            # Simplification: move toward the centroid of points that are "responsible" for the worst facets
            # Better: use the gradient from soft-max to get a direction
            soft_r, grads = _soft_max_radius_and_grad(current_pts, beta=200.0)
            
            if np.linalg.norm(grads[i]) > 1e-15:
                direction = grads[i] / np.linalg.norm(grads[i])
                new_pts[i] = current_pts[i] + 0.1 * direction
        
        new_pts = _normalize(new_pts)
        new_radius = _compute_covering_radius(new_pts)
        
        if new_radius < best_radius - 1e-14:
            best_radius = new_radius
            best_pts = new_pts.copy()
            current_pts = new_pts
    
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
    # Icosahedron has 12 vertices; we can start with those and add points
    phi_golden = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1]
    ], dtype=float)
    ico = _normalize(ico)
    seeds.append(ico[:12])  # Just the icosahedron as a starting point (will be padded)
    
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
    # 2. Electrostatic relaxation on top candidates
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
        
        budget = min(8.0, remaining / (n_seeds - i + 1))
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
    
    # Pass 2: Electrostatic relaxation on top 5 candidates
    top_candidates = results[:5]
    
    elapsed = time.time() - start_time
    remaining_for_pass2 = deadline - elapsed - 25.0  # Reserve 25s for final stages
    
    if remaining_for_pass2 > 10:
        time_per_seed_pass2 = remaining_for_pass2 / len(top_candidates)
        
        for rad, pts, idx in top_candidates:
            if time.time() > deadline - 25:
                break
                
            # Electrostatic relaxation
            relaxed_pts = _electrostatic_relax(pts, time_per_seed_pass2 * 0.5, power=4.0, penalty_weight=0.1)
            relaxed_rad = _compute_covering_radius(relaxed_pts)
            
            if relaxed_rad < best_radius:
                best_radius = relaxed_rad
                best_points = relaxed_pts.copy()
            
            # Worst facet optimization
            if time.time() < deadline - 25:
                worst_opt_pts = _worst_facet_optimization(relaxed_pts, time_per_seed_pass2 * 0.5)
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
