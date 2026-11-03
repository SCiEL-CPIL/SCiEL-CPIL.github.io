# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 300

def _hex_init(n_points, seed=0):
    """Generate hexagonal lattice points, select n_points closest to origin."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0
    
    for attempt in range(8):
        # Vary spacing to find good initial configuration
        spacing = 2.0 / (np.sqrt(n_points) + 1.0) * (0.82 + 0.04 * attempt)
        k = int(1.0 / spacing) + 4
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x**2 + y**2 <= (1.0 - spacing * 0.55)**2:
                    points.append([x, y])
        
        if len(points) < n_points:
            continue
        
        points = np.array(points)
        # Sort by distance from center
        radii = np.linalg.norm(points, axis=1)
        idx = np.argsort(radii)[:n_points]
        pts = points[idx]
        
        # Compute initial radius
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts
    
    if best_pts is None:
        # Fallback: uniform grid
        n = int(np.sqrt(n_points)) + 5
        xs = np.linspace(-0.95, 0.95, n)
        ys = np.linspace(-0.95, 0.95, n)
        X, Y = np.meshgrid(xs, ys)
        pts = np.column_stack([X.ravel(), Y.ravel()])
        mask = X.ravel()**2 + Y.ravel()**2 <= 0.95**2
        pts = pts[mask]
        if len(pts) > n_points:
            radii = np.linalg.norm(pts, axis=1)
            idx = np.argsort(radii)[:n_points]
            pts = pts[idx]
        best_pts = pts
    
    return best_pts

def _compute_radius(pts):
    """Compute the minimum radius for given point configuration."""
    n = len(pts)
    # Boundary constraint
    radii = np.linalg.norm(pts, axis=1)
    r_bound = 1.0 - np.max(radii)
    
    if r_bound <= 0:
        return 0.0
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r_pair = np.min(dist) / 2.0
    
    return min(r_bound, r_pair)

def _soft_min_logsumexp(values, beta):
    """Compute soft minimum using log-sum-exp for numerical stability."""
    min_val = np.min(values)
    if min_val > 1e10:
        return min_val
    shifted = -beta * (values - min_val)
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = min_val - log_sum / beta
    return soft_min

def _objective_and_grad(x, beta):
    """Objective: maximize soft minimum of (pair distances/2, boundary distances)."""
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist = np.sqrt(dist_sq + 1e-12)
    np.fill_diagonal(dist, np.inf)
    
    # Pair constraints: dist[i,j] / 2
    iu = np.triu_indices(n, 1)
    pair_vals = dist[iu] / 2.0
    
    # Boundary constraints: 1 - |pts[i]|
    radii = np.linalg.norm(pts, axis=1)
    boundary_vals = 1.0 - radii
    
    # Soft minimum of all constraints
    all_vals = np.concatenate([pair_vals, boundary_vals])
    soft_min = _soft_min_logsumexp(all_vals, beta)
    
    # Objective to minimize: -soft_min
    obj = -soft_min
    
    # Gradient computation
    min_val = np.min(all_vals)
    shifted = -beta * (all_vals - min_val)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    
    # Gradient for pairs
    grad = np.zeros_like(pts)
    
    pair_weights = weights[:len(pair_vals)]
    boundary_weights = weights[len(pair_vals):]
    
    # Create distance matrix with safe division
    dist_safe = dist.copy()
    dist_safe[dist_safe < 1e-8] = 1e-8
    
    # Unit vectors
    u = diff / dist_safe[:, :, np.newaxis]
    
    # Process pairs using vectorized operations
    idx_i, idx_j = iu
    # Only process pairs with significant weight
    active = pair_weights > 1e-10
    if np.any(active):
        ai, aj = idx_i[active], idx_j[active]
        aw = pair_weights[active]
        u_active = u[ai, aj]  # shape (num_active, 2)
        # Accumulate gradients
        np.add.at(grad, ai, aw[:, np.newaxis] * u_active)
        np.add.at(grad, aj, -aw[:, np.newaxis] * u_active)
    
    # Boundary contributions
    radii_safe = np.maximum(radii, 1e-8)
    active_b = boundary_weights > 1e-10
    if np.any(active_b):
        ib = np.where(active_b)[0]
        wb = boundary_weights[ib]
        grad[ib] -= wb[:, np.newaxis] * pts[ib] / radii_safe[ib, np.newaxis]
    
    # Total gradient
    total_grad = -grad  # Since obj = -soft_min
    
    return obj, total_grad.ravel()

def _voronoi_area_objective_and_grad(x, beta=10.0):
    """
    Objective based on Voronoi cell areas.
    Each point's 'value' is the area of its Voronoi cell (approximated).
    We maximize the soft-min of these areas, which encourages uniform spacing.
    Combined with boundary constraints.
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    # Approximate Voronoi cell area using nearest neighbor distances
    # For each point, compute distance to its k-th nearest neighbor
    # Area ~ (dist_to_kth_nn)^2 * constant
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist = np.sqrt(dist_sq + 1e-12)
    np.fill_diagonal(dist, np.inf)
    
    # For each point, find 4th nearest neighbor distance
    # This approximates the local density
    sorted_dists = np.sort(dist, axis=1)
    nn4_dists = sorted_dists[:, 3]  # 4th nearest (index 3)
    
    # Approximate Voronoi cell area
    # For hexagonal packing, area per point ~ (2r)^2 * sqrt(3)/2 where r is half-distance
    # So area ~ nn4_dist^2 * 0.866
    voronoi_areas = nn4_dists**2 * 0.866
    
    # Boundary constraint: 1 - |pts[i]|
    radii = np.linalg.norm(pts, axis=1)
    boundary_vals = 1.0 - radii
    
    # Scale voronoi areas to be comparable to boundary values
    # Target area for N=300 in unit circle: pi / 300 ~ 0.0105
    # But we want to maximize min area, so normalize
    target_area = np.pi / n
    normalized_areas = voronoi_areas / target_area
    
    # Combine: we want both large areas and good boundary
    # Use soft-min of normalized areas and boundary values
    all_vals = np.concatenate([normalized_areas, boundary_vals])
    soft_min = _soft_min_logsumexp(all_vals, beta)
    
    obj = -soft_min
    
    # Gradient
    min_val = np.min(all_vals)
    shifted = -beta * (all_vals - min_val)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    
    area_weights = weights[:n]
    boundary_weights = weights[n:]
    
    grad = np.zeros_like(pts)
    
    # Gradient of voronoi area w.r.t. pts
    # d(area_i)/d(pts_i) involves derivatives of nn distances
    # This is complex; use finite difference approximation or skip
    # For now, use a simpler approach: gradient of -soft_min of areas
    # The gradient of area_i w.r.t. pts_i is approximately proportional to 
    # the force from neighbors
    
    # Simplified: use the same pair-based gradient but weighted by area weights
    dist_safe = dist.copy()
    dist_safe[dist_safe < 1e-8] = 1e-8
    u = diff / dist_safe[:, :, np.newaxis]
    
    iu = np.triu_indices(n, 1)
    idx_i, idx_j = iu
    # Weight each pair by the min of the two area weights
    pair_w = np.minimum(area_weights[idx_i], area_weights[idx_j])
    active = pair_w > 1e-10
    if np.any(active):
        ai, aj = idx_i[active], idx_j[active]
        aw = pair_w[active]
        u_active = u[ai, aj]
        np.add.at(grad, ai, aw[:, np.newaxis] * u_active)
        np.add.at(grad, aj, -aw[:, np.newaxis] * u_active)
    
    # Boundary
    radii_safe = np.maximum(radii, 1e-8)
    active_b = boundary_weights > 1e-10
    if np.any(active_b):
        ib = np.where(active_b)[0]
        wb = boundary_weights[ib]
        grad[ib] -= wb[:, np.newaxis] * pts[ib] / radii_safe[ib, np.newaxis]
    
    total_grad = -grad
    
    return obj, total_grad.ravel()

def _simulated_annealing(pts, start_temp=0.01, end_temp=0.001, n_iter=5000, seed=42):
    """Simulated annealing to escape local minima."""
    rng = np.random.default_rng(seed)
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    current_pts = pts.copy()
    current_r = best_r
    
    for i in range(n_iter):
        temp = start_temp * (end_temp / start_temp) ** (i / n_iter)
        
        # Random perturbation
        n = len(current_pts)
        # Perturb a random subset of points
        n_perturb = max(1, n // 10)
        idx = rng.choice(n, size=n_perturb, replace=False)
        perturbations = rng.normal(0, temp, size=(n_perturb, 2))
        
        new_pts = current_pts.copy()
        new_pts[idx] += perturbations
        
        # Ensure inside unit circle
        radii = np.linalg.norm(new_pts, axis=1)
        mask = radii > 1.0
        if np.any(mask):
            new_pts[mask] = new_pts[mask] / radii[mask, np.newaxis] * 0.9999
        
        new_r = _compute_radius(new_pts)
        
        # Accept or reject
        if new_r > current_r:
            current_pts = new_pts
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts.copy()
        else:
            delta = current_r - new_r
            accept_prob = np.exp(-delta / max(temp, 1e-10))
            if rng.random() < accept_prob:
                current_pts = new_pts
                current_r = new_r
    
    return best_pts, best_r

def _tighten_min_constraint(pts, target_r, max_iter=100):
    """Try to push points to achieve at least target_r."""
    n = len(pts)
    pts = pts.copy()
    
    for _ in range(max_iter):
        # Find most violated constraint
        radii = np.linalg.norm(pts, axis=1)
        boundary_viol = target_r - (1.0 - radii)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        pair_viol = target_r - dist / 2.0
        
        # Find max violation
        iu = np.triu_indices(n, 1)
        pair_viol_mat = pair_viol[iu]
        max_pair_idx = np.argmax(pair_viol_mat)
        max_pair_viol = pair_viol_mat[max_pair_idx]
        
        max_bound_idx = np.argmax(boundary_viol)
        max_bound_viol = boundary_viol[max_bound_idx]
        
        if max_pair_viol < max_bound_viol and max_pair_viol < 1e-8:
            break
        
        if max_pair_viol >= max_bound_viol:
            # Push apart the most violated pair
            i, j = iu[0][max_pair_idx], iu[1][max_pair_idx]
            d = dist[i, j]
            if d < 1e-10:
                continue
            direction = (pts[i] - pts[j]) / d
            push = 0.1 * max_pair_viol
            pts[i] += direction * push
            pts[j] -= direction * push
        else:
            # Push inward the most violated boundary
            i = max_bound_idx
            r = radii[i]
            if r < 1e-10:
                continue
            direction = -pts[i] / r
            push = 0.1 * max_bound_viol
            pts[i] += direction * push
    
    # Ensure validity
    radii = np.linalg.norm(pts, axis=1)
    mask = radii > 1.0
    if np.any(mask):
        pts[mask] = pts[mask] / radii[mask, np.newaxis] * 0.9999
    
    return pts

def _local_refine(pts, beta, maxiter=500):
    """Local refinement using L-BFGS-B."""
    bounds = [(-1.0, 1.0)] * (2 * len(pts))
    
    result = minimize(
        _objective_and_grad,
        pts.ravel(),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        args=(beta,),
        options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30}
    )
    
    best_pts = result.x.reshape(-1, 2)
    
    # Ensure validity
    radii = np.linalg.norm(best_pts, axis=1)
    mask = radii > 1.0
    if np.any(mask):
        best_pts[mask] = best_pts[mask] / radii[mask, np.newaxis] * 0.9999
    
    return best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # 1. Initialization
    pts = _hex_init(N, seed=42)
    r_init = _compute_radius(pts)
    
    best_pts = pts
    best_r = r_init
    
    # 2. Simulated annealing to escape local minima
    if time.time() < deadline - 60.0:
        try:
            sa_pts, sa_r = _simulated_annealing(
                best_pts, start_temp=0.02, end_temp=0.001, n_iter=3000, seed=123
            )
            if sa_r > best_r:
                best_r = sa_r
                best_pts = sa_pts
        except Exception:
            pass
    
    # 3. Multi-scale optimization with L-BFGS-B
    beta_schedule = [50.0, 30.0, 20.0, 15.0, 10.0, 8.0, 6.0, 5.0]
    
    for beta in beta_schedule:
        if time.time() > deadline - 30.0:
            break
        
        try:
            pts_opt = _local_refine(best_pts, beta, maxiter=300)
            r_opt = _compute_radius(pts_opt)
            
            if r_opt > best_r:
                best_r = r_opt
                best_pts = pts_opt
        except Exception:
            continue
        
        if time.time() > deadline - 20.0:
            break
    
    # 4. Another round of simulated annealing if time permits
    if time.time() < deadline - 20.0:
        try:
            sa_pts, sa_r = _simulated_annealing(
                best_pts, start_temp=0.01, end_temp=0.0005, n_iter=2000, seed=456
            )
            if sa_r > best_r:
                best_r = sa_r
                best_pts = sa_pts
        except Exception:
            pass
    
    # 5. Final tightening
    if time.time() < deadline - 10.0:
        target_r = best_r * 1.001
        pts_tight = _tighten_min_constraint(best_pts, target_r, max_iter=50)
        r_tight = _compute_radius(pts_tight)
        
        if r_tight > best_r:
            best_r = r_tight
            best_pts = pts_tight
    
    # 6. Final local refinement
    if time.time() < deadline - 5.0:
        try:
            pts_final = _local_refine(best_pts, 10.0, maxiter=200)
            r_final = _compute_radius(pts_final)
            
            if r_final > best_r:
                best_r = r_final
                best_pts = pts_final
        except Exception:
            pass
    
    # Ensure final validity
    radii = np.linalg.norm(best_pts, axis=1)
    mask = radii > 1.0
    if np.any(mask):
        best_pts[mask] = best_pts[mask] / radii[mask, np.newaxis] * 0.9999
    
    # Verify shape
    assert best_pts.shape == (N, 2), f"Wrong shape: {best_pts.shape}"
    assert np.all(np.isfinite(best_pts)), "Non-finite values"
    
    return best_pts
# EVOLVE-BLOCK-END
