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
    
    for attempt in range(6):
        # Vary spacing to find good initial configuration
        spacing = 2.0 / (np.sqrt(n_points) + 1.0) * (0.85 + 0.05 * attempt)
        k = int(1.0 / spacing) + 4
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x**2 + y**2 <= (1.0 - spacing * 0.6)**2:
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
    # soft_min = -1/beta * log(sum(exp(-beta * x)))
    # Use log-sum-exp trick for stability
    min_val = np.min(values)
    if min_val > 1e10:
        return min_val
    shifted = -beta * (values - min_val)
    # Clip to avoid overflow
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
    # d(soft_min)/dx = -1/beta * sum_j exp(-beta*x_j) * dx_j/dx / sum_j exp(-beta*x_j)
    # Let's compute weights w_j = exp(-beta * x_j) / sum(exp(-beta*x_j))
    
    # For numerical stability
    min_val = np.min(all_vals)
    shifted = -beta * (all_vals - min_val)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    
    # Gradient for pairs
    grad = np.zeros_like(pts)
    
    # Pair contributions
    # For pair (i,j): x_ij = dist(i,j)/2
    # d(x_ij)/d(pts[i]) = (pts[i] - pts[j]) / dist(i,j)
    # d(x_ij)/d(pts[j]) = (pts[j] - pts[i]) / dist(i,j)
    
    # We need to accumulate: sum_j w_j * d(x_j)/dx
    # Let's iterate over pairs for clarity (N=300, ~45k pairs)
    
    pair_weights = weights[:len(pair_vals)]
    boundary_weights = weights[len(pair_vals):]
    
    # Process pairs
    # To be efficient, use matrix operations where possible
    # But for gradient, we need per-pair contributions
    
    # Create distance matrix with safe division
    dist_safe = dist.copy()
    dist_safe[dist_safe < 1e-8] = 1e-8
    
    # Unit vectors: u_ij = (pts[i] - pts[j]) / dist(i,j)
    # u[i,j] = diff[i,j] / dist[i,j]
    u = diff / dist_safe[:, :, np.newaxis]
    
    # For each pair (i,j) with i<j, weight w_ij
    # Contribution to grad[i]: w_ij * u[i,j]
    # Contribution to grad[j]: -w_ij * u[i,j]
    
    # Use einsum for efficiency
    # u has shape (n, n, 2)
    # pair_weights has shape (n*(n-1)/2,)
    # We need to map pair indices to (i,j) pairs
    
    # Simpler approach: loop over pairs (acceptable for N=300)
    idx_i, idx_j = iu
    for k in range(len(idx_i)):
        i, j = idx_i[k], idx_j[k]
        w = pair_weights[k]
        if w < 1e-12:
            continue
        u_ij = u[i, j]  # (pts[i] - pts[j]) / dist
        grad[i] += w * u_ij
        grad[j] -= w * u_ij
    
    # Boundary contributions
    # For point i: x_i = 1 - |pts[i]|
    # d(x_i)/d(pts[i]) = -pts[i] / |pts[i]|
    radii_safe = np.maximum(radii, 1e-8)
    for i in range(n):
        w = boundary_weights[i]
        if w < 1e-12:
            continue
        grad[i] -= w * pts[i] / radii_safe[i]
    
    # Total gradient
    total_grad = -grad  # Since obj = -soft_min
    
    return obj, total_grad.ravel()

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

def _local_refine(pts, beta, time_limit=5.0):
    """Local refinement using L-BFGS-B."""
    start = time.time()
    
    bounds = [(-1.0, 1.0)] * (2 * len(pts))
    
    result = minimize(
        _objective_and_grad,
        pts.ravel(),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        args=(beta,),
        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30}
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
    
    # 2. Multi-scale optimization
    best_pts = pts
    best_r = r_init
    
    # Start with high beta (sharp) and decrease
    beta_schedule = [50.0, 30.0, 20.0, 15.0, 10.0, 8.0, 6.0, 5.0]
    
    for beta in beta_schedule:
        if time.time() > deadline - 30.0:
            break
        
        # Optimize with current beta
        try:
            pts_opt = _local_refine(best_pts, beta, time_limit=15.0)
            r_opt = _compute_radius(pts_opt)
            
            if r_opt > best_r:
                best_r = r_opt
                best_pts = pts_opt
        except Exception:
            continue
        
        # Check time
        if time.time() > deadline - 20.0:
            break
    
    # 3. Final tightening
    if time.time() < deadline - 10.0:
        target_r = best_r * 1.001
        pts_tight = _tighten_min_constraint(best_pts, target_r, max_iter=50)
        r_tight = _compute_radius(pts_tight)
        
        if r_tight > best_r:
            best_r = r_tight
            best_pts = pts_tight
    
    # 4. Final local refinement
    if time.time() < deadline - 5.0:
        try:
            pts_final = _local_refine(best_pts, 10.0, time_limit=5.0)
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
