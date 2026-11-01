import numpy as np
import time
from scipy.optimize import minimize

N = 300

def _compute_score(points):
    """Compute the actual packing radius for given points."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Check if all points are inside unit circle
    radii = np.hypot(points[:, 0], points[:, 1])
    if np.any(radii > 1.0):
        return 0.0
    
    # Compute pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    min_wall_dist = np.min(1.0 - radii)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _soft_min_and_grad(points, beta=50.0):
    """
    Compute soft-min of all constraints (pair distances and wall distances)
    and its gradient. Uses log-sum-exp for numerical stability.
    
    Returns: (soft_min_value, gradient)
    """
    n = points.shape[0]
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Wall distances: 1 - |c_i|
    radii = np.hypot(points[:, 0], points[:, 1])
    wall_dists = 1.0 - radii
    
    # Combine all constraints into one vector
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu]
    wall = wall_dists
    
    all_constraints = np.concatenate([pair_dists, wall])
    
    # Soft-min via log-sum-exp
    neg_beta_c = -beta * all_constraints
    max_neg_beta_c = np.max(neg_beta_c)
    exp_terms = np.exp(neg_beta_c - max_neg_beta_c)
    sum_exp = np.sum(exp_terms)
    
    soft_min_val = -(np.log(sum_exp) + max_neg_beta_c) / beta
    
    # Gradient
    weights = exp_terms / sum_exp
    
    grad = np.zeros_like(points)
    
    # Gradient from pair constraints
    pair_weights = weights[:len(pair_dists)]
    i_idx, j_idx = iu
    safe_dists = np.where(pair_dists < 1e-10, 1e-10, pair_dists)
    
    dir_ij = (points[i_idx] - points[j_idx]) / safe_dists[:, np.newaxis]
    
    np.add.at(grad, i_idx, pair_weights[:, np.newaxis] * dir_ij)
    np.add.at(grad, j_idx, -pair_weights[:, np.newaxis] * dir_ij)
    
    # Gradient from wall constraints
    wall_weights = weights[len(pair_dists):]
    safe_radii = np.where(radii < 1e-10, 1e-10, radii)
    wall_grad = -wall_weights[:, np.newaxis] * points / safe_radii[:, np.newaxis]
    grad += wall_grad
    
    return soft_min_val, grad

def _hex_lattice_init():
    """Generate hexagonal lattice initialization."""
    r_init = 0.053
    spacing = 2.0 * r_init
    points = []
    k = int(1.0 / spacing) + 5
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            points.append([x, y])
    
    points = np.array(points)
    dists = np.hypot(points[:, 0], points[:, 1])
    points = points[dists <= 1.0 - r_init]
    
    if len(points) > N:
        dists = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(dists)
        points = points[idx[:N]]
    elif len(points) < N:
        extra = N - len(points)
        new_pts = np.random.uniform(-0.8, 0.8, (extra, 2))
        points = np.vstack((points, new_pts))
    
    points = points[:N]
    points += np.random.uniform(-1e-4, 1e-4, points.shape)
    return points

def _ring_init():
    """Generate concentric ring initialization."""
    points = []
    points.append([0.0, 0.0])
    
    ring = 1
    while len(points) < N and ring <= 15:
        n_in_ring = 6 * ring
        if len(points) + n_in_ring > N:
            n_in_ring = N - len(points)
        
        r_ring = ring * 0.108
        
        if r_ring > 0.92:
            break
            
        for i in range(n_in_ring):
            angle = 2.0 * np.pi * i / n_in_ring + 0.1 * ring
            x = r_ring * np.cos(angle)
            y = r_ring * np.sin(angle)
            points.append([x, y])
        
        ring += 1
    
    points = np.array(points)
    if len(points) < N:
        extra = N - len(points)
        new_pts = np.random.uniform(-0.8, 0.8, (extra, 2))
        points = np.vstack((points, new_pts))
    else:
        points = points[:N]
    
    points += np.random.uniform(-1e-4, 1e-4, points.shape)
    return points

def _local_optimize(points, beta_schedule, max_iter_per_phase=300, time_limit=None):
    """
    Local optimization using L-BFGS-B with increasing beta.
    """
    start_time = time.time()
    current_points = points.copy()
    
    for beta in beta_schedule:
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        def obj(x):
            p = x.reshape(-1, 2)
            val, _ = _soft_min_and_grad(p, beta)
            return -val
        
        def jac(x):
            p = x.reshape(-1, 2)
            _, grad = _soft_min_and_grad(p, beta)
            return -grad
        
        x0 = current_points.flatten()
        
        result = minimize(
            obj,
            x0,
            jac=jac,
            method='L-BFGS-B',
            options={'maxiter': max_iter_per_phase, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        current_points = result.x.reshape(-1, 2)
        
        radii = np.hypot(current_points[:, 0], current_points[:, 1])
        mask = radii > 1.0
        if np.any(mask):
            current_points[mask] /= radii[mask, np.newaxis]
    
    return current_points

def _push_optimize(points, target_r, time_limit=30.0):
    """
    Final push: optimize to achieve a specific target radius using penalty.
    """
    start_time = time.time()
    current_points = points.copy()
    
    def penalty_obj(x):
        p = x.reshape(-1, 2)
        
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        radii = np.hypot(p[:, 0], p[:, 1])
        
        pair_viol = np.maximum(0, 2.0 * target_r - dists)
        wall_viol = np.maximum(0, radii - (1.0 - target_r))
        
        penalty = np.sum(pair_viol**2) + np.sum(wall_viol**2)
        
        min_pair = np.min(dists)
        min_wall = np.min(1.0 - radii)
        min_constraint = min(min_pair / 2.0, min_wall)
        
        return penalty - 0.01 * min_constraint
    
    def penalty_jac(x):
        p = x.reshape(-1, 2)
        
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        dists = np.sqrt(dists_sq)
        
        radii = np.hypot(p[:, 0], p[:, 1])
        
        pair_viol = np.maximum(0, 2.0 * target_r - dists)
        wall_viol = np.maximum(0, radii - (1.0 - target_r))
        
        grad = np.zeros_like(p)
        
        safe_dists = np.where(dists < 1e-10, 1e-10, dists)
        unit_vec = diff / safe_dists[:, :, np.newaxis]
        pair_grad = -2.0 * pair_viol[:, :, np.newaxis] * unit_vec
        grad += np.sum(pair_grad, axis=1)
        
        safe_radii = np.where(radii < 1e-10, 1e-10, radii)
        wall_grad = 2.0 * wall_viol[:, np.newaxis] * (p / safe_radii[:, np.newaxis])
        grad += wall_grad
        
        return grad
    
    x0 = current_points.flatten()
    
    result = minimize(
        penalty_obj,
        x0,
        jac=penalty_jac,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
    )
    
    current_points = result.x.reshape(-1, 2)
    
    radii = np.hypot(current_points[:, 0], current_points[:, 1])
    mask = radii > 1.0
    if np.any(mask):
        current_points[mask] /= radii[mask, np.newaxis]
    
    return current_points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_points = None
    best_score = 0.0
    
    inits = [
        _hex_lattice_init(),
        _ring_init()
    ]
    
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline:
            break
        
        time_remaining = deadline - time.time()
        if time_remaining < 20:
            break
        
        beta_schedule = [10.0, 25.0, 50.0, 100.0, 150.0]
        
        optimized_pts = _local_optimize(
            init_pts,
            beta_schedule,
            max_iter_per_phase=200,
            time_limit=time_remaining * 0.5
        )
        
        current_score = _compute_score(optimized_pts)
        
        time_remaining = deadline - time.time()
        if time_remaining < 10:
            break
        
        diff = optimized_pts[:, np.newaxis, :] - optimized_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        radii = np.hypot(optimized_pts[:, 0], optimized_pts[:, 1])
        min_wall_dist = np.min(1.0 - radii)
        current_r = min(min_pair_dist / 2.0, min_wall_dist)
        
        target_r = current_r * 1.002
        pushed_pts = _push_optimize(
            optimized_pts,
            target_r,
            time_limit=time_remaining * 0.4
        )
        
        time_remaining = deadline - time.time()
        if time_remaining < 5:
            break
        
        final_beta_schedule = [100.0, 200.0, 300.0]
        polished_pts = _local_optimize(
            pushed_pts,
            final_beta_schedule,
            max_iter_per_phase=150,
            time_limit=time_remaining * 0.7
        )
        
        score = _compute_score(polished_pts)
        
        if score > best_score:
            best_score = score
            best_points = polished_pts.copy()
    
    if best_points is None:
        best_points = inits[0]
    
    radii = np.hypot(best_points[:, 0], best_points[:, 1])
    mask = radii >= 1.0
    if np.any(mask):
        best_points[mask] *= 0.999
    
    return best_points
