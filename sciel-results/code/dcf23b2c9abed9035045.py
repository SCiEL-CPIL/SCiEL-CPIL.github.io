"""Packing 200 equal circles in a unit square: Multi-temperature L-BFGS-B + Active Set Polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

N = 200
DEADLINE = 142.0  # seconds; leave margin before 150s kill

def _score(pts):
    """Return the common radius r for a valid configuration, or 0.0 if invalid."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Distance to boundary
    ax = np.abs(pts[:, 0])
    ay = np.abs(pts[:, 1])
    boundary = np.minimum(0.5 - ax, 0.5 - ay)
    min_boundary = np.min(boundary)
    if min_boundary < -1e-10:
        return 0.0
    # Pairwise distances
    dist = cdist(pts, pts)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    r = min(min_boundary, min_dist / 2.0)
    return r if r > 0 else 0.0

def _hex_init(jitter_scale=0.0, seed=0):
    """Generate a hexagonal lattice initialization with multiple aspect ratios."""
    best_pts = None
    best_r = 0
    rng = np.random.default_rng(seed)
    
    # Try a wider range of spacings and aspect ratios
    s_est = 1.0 / np.sqrt(N) * 1.2
    for s in np.linspace(s_est * 0.85, s_est * 1.15, 25):
        for aspect in [1.0, 0.95, 1.05, 0.9, 1.1]:
            pts_list = []
            row = 0
            while True:
                y = -0.5 + s/2 + row * s * np.sqrt(3) / 2 * aspect
                if y > 0.5 - s/2:
                    break
                offset = (row % 2) * s / 2
                x = -0.5 + s/2 + offset
                while x <= 0.5 - s/2:
                    pts_list.append((x, y))
                    x += s
                row += 1
            if len(pts_list) < N:
                continue
            pts = np.array(pts_list[:N])
            if jitter_scale > 0:
                pts += rng.normal(0, jitter_scale, size=pts.shape)
                pts = np.clip(pts, -0.49, 0.49)
            r = _score(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
    
    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        pts = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = pts[:N].copy()
    return best_pts

def _random_init(seed=0, n_clusters=10):
    """Generate a random clustered initialization for diversity."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-0.45, 0.45, size=(N, 2))
    # Add some structure by clustering
    cluster_centers = rng.uniform(-0.4, 0.4, size=(n_clusters, 2))
    for i in range(N):
        c = i % n_clusters
        pts[i] = pts[i] * 0.5 + cluster_centers[c] * 0.5
    pts = np.clip(pts, -0.49, 0.49)
    return pts

def _force_directed(pts, time_budget):
    """
    Force-directed relaxation: repulsion between points, attraction to center/bounds.
    Uses KDTree for efficient neighbor lookup to speed up iterations.
    """
    t0 = time.time()
    pts = pts.copy()
    current_r = _score(pts)
    
    # Parameters
    learning_rate = 0.008
    min_lr = 1e-7
    repulsion_power = 1.5
    boundary_repulsion = 2.0
    neighbor_k = 15  # only consider nearest neighbors for speed
    
    while time.time() - t0 < time_budget:
        forces = np.zeros_like(pts)
        
        # Build KDTree for efficient neighbor queries
        tree = cKDTree(pts)
        dists, indices = tree.query(pts, k=neighbor_k + 1)
        
        # Pairwise repulsion from nearest neighbors only
        for i in range(N):
            for j_idx in range(1, neighbor_k + 1):
                j = int(indices[i, j_idx])
                d = dists[i, j_idx]
                if d < 1e-10:
                    continue
                # Force magnitude: stronger for closer points
                weight = 1.0 / (d ** repulsion_power)
                diff = pts[i] - pts[j]
                forces[i] += diff / d * weight
        
        # Boundary repulsion - push points away from edges
        d_x = 0.5 - np.abs(pts[:, 0])
        d_y = 0.5 - np.abs(pts[:, 1])
        boundary_dists = np.minimum(d_x, d_y)
        
        r_est = max(current_r, 0.001)
        close_to_boundary = boundary_dists < r_est * 1.5
        if np.any(close_to_boundary):
            # Push inward
            push = -pts * boundary_repulsion
            forces[close_to_boundary] += push[close_to_boundary]
        
        # Normalize forces to prevent explosion
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        max_norm = 0.5
        too_large = force_norms[:, 0] > max_norm
        if np.any(too_large):
            forces[too_large] *= (max_norm / force_norms[too_large, 0])[:, np.newaxis]
        
        # Update positions
        pts += learning_rate * forces
        pts = np.clip(pts, -0.499, 0.499)
        
        new_r = _score(pts)
        if new_r > current_r:
            current_r = new_r
            learning_rate *= 0.995
        else:
            learning_rate *= 0.97
        
        if learning_rate < min_lr:
            break
            
    return pts

def _soft_min_neg(x, alpha):
    """
    Objective function for L-BFGS-B using numerically stable log-sum-exp.
    x: flat array of size 2N.
    alpha: sharpness of the soft-min.
    Returns: -soft_min(distances)
    """
    pts = x.reshape(N, 2)
    
    # Distances to boundary
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    boundary_dists = np.minimum(d_x, d_y).flatten()
    
    # Pairwise distances
    dist = cdist(pts, pts)
    np.fill_diagonal(dist, np.inf)
    pairwise_dists = dist[~np.isinf(dist)]
    
    # Combine all distances
    all_dists = np.concatenate([boundary_dists, pairwise_dists])
    
    # Numerically stable log-sum-exp for soft min
    # soft_min = -1/alpha * log(sum(exp(-alpha * d)))
    # Use max trick for numerical stability
    min_d = np.min(all_dists)
    exp_terms = np.exp(-alpha * (all_dists - min_d))
    log_sum = np.log(np.sum(exp_terms)) + (-alpha * min_d)
    soft_min_val = -log_sum / alpha
    
    # We want to maximize min distance, so minimize -min distance.
    return -soft_min_val

def _soft_min_neg_grad(x, alpha):
    """
    Objective + analytic gradient for L-BFGS-B.
    Returns (value, grad) where grad is d(-soft_min)/dx.
    """
    pts = x.reshape(N, 2)
    
    # Distances to boundary
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    boundary_dists = np.minimum(d_x, d_y)
    
    # Pairwise distances
    dist = cdist(pts, pts)
    np.fill_diagonal(dist, np.inf)
    pairwise_dists = dist[~np.isinf(dist)]
    
    # Combine all distances
    all_dists = np.concatenate([boundary_dists.flatten(), pairwise_dists])
    
    # Soft-min weights: w_k = exp(-alpha * d_k) / sum(exp(-alpha * d_j))
    min_d = np.min(all_dists)
    exp_terms = np.exp(-alpha * (all_dists - min_d))
    weights = exp_terms / np.sum(exp_terms)
    
    # soft_min = -1/alpha * log(sum(exp(-alpha*d)))
    # d(soft_min)/d(x_i) = sum_k w_k * d(d_k)/d(x_i)
    # We minimize -soft_min, so grad = -d(soft_min)/dx = -sum_k w_k * d(d_k)/dx
    
    grad = np.zeros(2 * N)
    
    # Boundary distance contributions
    # d_x = 0.5 - |x_i|, d_y = 0.5 - |y_i|
    # boundary_dist_i = min(d_x_i, d_y_i)
    # d(boundary_dist_i)/dx_i = -sign(x_i) if d_x_i <= d_y_i else 0
    # d(boundary_dist_i)/dy_i = -sign(y_i) if d_y_i < d_x_i else 0
    
    w_boundary = weights[:N]
    for i in range(N):
        if d_x[i] <= d_y[i]:
            # boundary dist = d_x, derivative wrt x is -sign(x_i)
            sx = np.sign(pts[i, 0])
            if sx == 0:
                sx = 1.0
            grad[2*i] += w_boundary[i] * (-sx)
        else:
            # boundary dist = d_y, derivative wrt y is -sign(y_i)
            sy = np.sign(pts[i, 1])
            if sy == 0:
                sy = 1.0
            grad[2*i+1] += w_boundary[i] * (-sy)
    
    # Pairwise distance contributions
    # d(|c_i - c_j|)/d(c_i) = (c_i - c_j) / |c_i - c_j|
    # d(|c_i - c_j|)/d(c_j) = (c_j - c_i) / |c_i - c_j|
    w_pair = weights[N:]
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            if w_pair[idx] < 1e-15:
                idx += 1
                continue
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                idx += 1
                continue
            unit = diff / d
            # d(dist_ij)/d(c_i) = unit, so d(-soft_min)/d(c_i) = -w * unit
            grad[2*i] += -w_pair[idx] * unit[0]
            grad[2*i+1] += -w_pair[idx] * unit[1]
            grad[2*j] += w_pair[idx] * unit[0]
            grad[2*j+1] += w_pair[idx] * unit[1]
            idx += 1
    
    min_d = np.min(all_dists)
    exp_terms = np.exp(-alpha * (all_dists - min_d))
    log_sum = np.log(np.sum(exp_terms)) + (-alpha * min_d)
    soft_min_val = -log_sum / alpha
    
    return -soft_min_val, grad

def _optimize_lbfgs(pts, time_budget):
    """
    Run L-BFGS-B optimization with multi-temperature annealing schedule.
    Uses analytic gradient for much faster convergence.
    """
    t0 = time.time()
    x0 = pts.flatten()
    
    # Multi-temperature schedule: coarse to fine
    phases = [
        {"alpha": 8.0, "max_iter": 150, "gtol": 1e-3},
        {"alpha": 25.0, "max_iter": 200, "gtol": 1e-5},
        {"alpha": 80.0, "max_iter": 250, "gtol": 1e-6},
        {"alpha": 250.0, "max_iter": 250, "gtol": 1e-7},
        {"alpha": 800.0, "max_iter": 200, "gtol": 1e-8},
        {"alpha": 2000.0, "max_iter": 150, "gtol": 1e-9},
    ]
    
    for phase in phases:
        if time.time() - t0 > time_budget:
            break
            
        alpha = phase["alpha"]
        
        def obj(x):
            return _soft_min_neg(x, alpha)
        
        def obj_grad(x):
            return _soft_min_neg_grad(x, alpha)
        
        try:
            res = minimize(obj_grad, x0, method='L-BFGS-B', 
                           bounds=[(-0.4999, 0.4999)] * (2 * N),
                           options={'maxiter': phase["max_iter"], 'gtol': phase["gtol"],
                                    'ftol': 1e-12, 'maxls': 50})
            x_opt = res.x
            x0 = x_opt
        except Exception:
            pass
            
    return x0.reshape(N, 2)

def _active_set_polish(pts, time_budget):
    """
    High-precision polish: only move points involved in the minimum distance constraints.
    Uses pairwise separation moves and directed perturbations.
    """
    t0 = time.time()
    best_pts = pts.copy()
    best_r = _score(best_pts)
    
    rng = np.random.default_rng(54321)
    step_size = 0.002
    min_step = 1e-7
    stagnation_count = 0
    
    while time.time() - t0 < time_budget:
        # Compute constraints
        d_x = 0.5 - np.abs(best_pts[:, 0])
        d_y = 0.5 - np.abs(best_pts[:, 1])
        boundary_dists = np.minimum(d_x, d_y)
        min_boundary = np.min(boundary_dists)
        
        dist = cdist(best_pts, best_pts)
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist)
        
        current_r = min(min_boundary, min_pair / 2.0)
        
        # Identify active set
        active_idx = set()
        
        tight_boundary = np.where(boundary_dists < current_r * 1.05)[0]
        active_idx.update(tight_boundary.tolist())
        
        # Find tight pairs using KDTree
        tree = cKDTree(best_pts)
        pairs, dists = tree.query(best_pts, k=5)
        tight_pairs = []
        for i in range(N):
            for j in range(1, 5):
                if dists[i, j] < 2.0 * current_r * 1.05:
                    active_idx.add(i)
                    active_idx.add(int(pairs[i, j]))
                    tight_pairs.append((i, int(pairs[i, j]), dists[i, j]))
        
        if not active_idx:
            break
            
        active_list = list(active_idx)
        n_active = len(active_list)
        
        # Try perturbing active points with multiple strategies
        improved = False
        
        # Strategy 1: Pairwise separation - push close pairs apart simultaneously
        if not improved and tight_pairs:
            for _ in range(10):
                # Pick a random tight pair
                pi, pj, pd = tight_pairs[rng.integers(len(tight_pairs))]
                diff = best_pts[pi] - best_pts[pj]
                norm = np.linalg.norm(diff)
                if norm < 1e-10:
                    continue
                unit = diff / norm
                # Push both apart
                perturb = np.zeros((N, 2))
                perturb[pi] = unit * step_size * 0.5
                perturb[pj] = -unit * step_size * 0.5
                trial = best_pts + perturb
                trial = np.clip(trial, -0.4999, 0.4999)
                trial_r = _score(trial)
                if trial_r > best_r:
                    best_pts = trial
                    best_r = trial_r
                    improved = True
                    break
        
        # Strategy 2: Random perturbations of active set
        if not improved:
            for _ in range(10):
                perturb = np.zeros((N, 2))
                perturb[active_list] = rng.normal(0, step_size, size=(n_active, 2))
                trial = best_pts + perturb
                trial = np.clip(trial, -0.4999, 0.4999)
                trial_r = _score(trial)
                if trial_r > best_r:
                    best_pts = trial
                    best_r = trial_r
                    improved = True
                    break
        
        # Strategy 3: Directed perturbations away from nearest neighbors
        if not improved:
            for i in active_list:
                nearest_dist, nearest_idx = tree.query(best_pts[i], k=2)
                for j_idx in range(1, 2):
                    j = int(nearest_idx[j_idx])
                    if j == i:
                        continue
                    diff = best_pts[i] - best_pts[j]
                    norm = np.linalg.norm(diff)
                    if norm > 1e-10:
                        perturb = np.zeros((N, 2))
                        perturb[i] = diff / norm * step_size
                        trial = best_pts + perturb
                        trial = np.clip(trial, -0.4999, 0.4999)
                        trial_r = _score(trial)
                        if trial_r > best_r:
                            best_pts = trial
                            best_r = trial_r
                            improved = True
                            break
                if improved:
                    break
        
        if improved:
            step_size *= 0.95
            stagnation_count = 0
        else:
            step_size *= 0.85
            stagnation_count += 1
            if step_size < min_step or stagnation_count > 50:
                break
                
    return best_pts

def _high_precision_polish(pts, time_budget):
    """
    Very high-precision polish using a smooth surrogate with very high alpha
    and tight tolerances, focusing on the final digits of precision.
    Uses analytic gradient for speed.
    """
    t0 = time.time()
    x0 = pts.flatten()
    
    # Use very high alpha for near-exact min approximation
    phases = [
        {"alpha": 5000.0, "max_iter": 100, "gtol": 1e-10},
        {"alpha": 15000.0, "max_iter": 80, "gtol": 1e-11},
        {"alpha": 40000.0, "max_iter": 60, "gtol": 1e-12},
    ]
    
    for phase in phases:
        if time.time() - t0 > time_budget:
            break
            
        alpha = phase["alpha"]
        
        def obj(x):
            return _soft_min_neg(x, alpha)
        
        def obj_grad(x):
            return _soft_min_neg_grad(x, alpha)
        
        try:
            res = minimize(obj_grad, x0, method='L-BFGS-B', 
                           bounds=[(-0.49999, 0.49999)] * (2 * N),
                           options={'maxiter': phase["max_iter"], 'gtol': phase["gtol"],
                                    'ftol': 1e-14, 'maxls': 30})
            x_opt = res.x
            x0 = x_opt
        except Exception:
            pass
            
    return x0.reshape(N, 2)

def construct():
    t_start = time.time()
    
    # 1. Initialization - try diverse starting points
    best_pts = None
    best_r = 0.0
    
    inits = [
        _hex_init(jitter_scale=0.0, seed=0),
        _hex_init(jitter_scale=0.003, seed=1),
        _hex_init(jitter_scale=0.008, seed=2),
        _hex_init(jitter_scale=0.015, seed=3),
        _random_init(seed=42),
        _random_init(seed=123),
    ]
    
    for pts in inits:
        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
            
    # 2. Force-Directed Relaxation - faster with KDTree (shorter budget since gradient-based L-BFGS is faster)
    remaining = DEADLINE - time.time()
    if remaining > 12:
        fd_budget = min(remaining * 0.12, 10.0)
        fd_pts = _force_directed(best_pts, fd_budget)
        fd_r = _score(fd_pts)
        if fd_r > best_r:
            best_r = fd_r
            best_pts = fd_pts.copy()
            
    # 3. L-BFGS-B Polish with multi-temperature annealing (now with analytic gradient)
    remaining = DEADLINE - time.time()
    if remaining > 20:
        lbfgs_budget = min(remaining * 0.45, 55.0)
        opt_pts = _optimize_lbfgs(best_pts, lbfgs_budget)
        opt_pts = np.clip(opt_pts, -0.4999, 0.4999)
        opt_r = _score(opt_pts)
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()
            
    # 4. Active Set Polish
    remaining = DEADLINE - time.time()
    if remaining > 8:
        polish_budget = min(remaining * 0.2, 15.0)
        polished_pts = _active_set_polish(best_pts, polish_budget)
        polished_r = _score(polished_pts)
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished_pts.copy()
            
    # 5. Second L-BFGS-B pass with fresh multi-temperature schedule
    remaining = DEADLINE - time.time()
    if remaining > 18:
        lbfgs_budget = min(remaining * 0.4, 35.0)
        opt_pts = _optimize_lbfgs(best_pts, lbfgs_budget)
        opt_pts = np.clip(opt_pts, -0.4999, 0.4999)
        opt_r = _score(opt_pts)
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()
            
    # 6. Active Set Polish again
    remaining = DEADLINE - time.time()
    if remaining > 8:
        polish_budget = min(remaining * 0.25, 12.0)
        polished_pts = _active_set_polish(best_pts, polish_budget)
        polished_r = _score(polished_pts)
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished_pts.copy()
            
    # 7. High-precision polish with very high alpha
    remaining = DEADLINE - time.time()
    if remaining > 10:
        hp_budget = min(remaining * 0.45, 28.0)
        hp_pts = _high_precision_polish(best_pts, hp_budget)
        hp_pts = np.clip(hp_pts, -0.49999, 0.49999)
        hp_r = _score(hp_pts)
        if hp_r > best_r:
            best_r = hp_r
            best_pts = hp_pts.copy()
            
    # 8. Final active set polish
    remaining = DEADLINE - time.time()
    if remaining > 5:
        polish_budget = min(remaining * 0.5, 10.0)
        polished_pts = _active_set_polish(best_pts, polish_budget)
        polished_r = _score(polished_pts)
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished_pts.copy()
            
    # 9. Final safety check
    if best_pts is None:
        best_pts = _hex_init()
        
    # Ensure all points are strictly inside the square
    best_pts = np.clip(best_pts, -0.49999, 0.49999)
    
    return best_pts
