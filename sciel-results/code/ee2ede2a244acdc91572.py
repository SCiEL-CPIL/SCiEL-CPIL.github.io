"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the packing radius for given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    dist_x = 0.5 - np.abs(centers[:, 0])
    dist_y = 0.5 - np.abs(centers[:, 1])
    min_bound = np.min(np.minimum(dist_x, dist_y))
    
    return min(min_pair, min_bound)

def _epigraph_obj(x, n):
    """
    Epigraph formulation: x = [centers (2n), t]
    Maximize t subject to all pairwise distances >= 2t and boundary >= t.
    We minimize -t + penalty for violations.
    """
    centers = x[:2*n].reshape(n, 2)
    t = x[2*n]
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    # Boundary distances
    dist_x = 0.5 - np.abs(centers[:, 0])
    dist_y = 0.5 - np.abs(centers[:, 1])
    bound_vals = np.minimum(dist_x, dist_y)
    
    # Violations: how much constraints are violated
    # For pairs: violation = 2t - dist_ij (positive if violated)
    # For boundary: violation = t - b_i
    pair_viol = 2.0 * t - dists
    bound_viol = t - bound_vals
    
    # Soft penalty (smooth)
    penalty = np.sum(np.maximum(0, pair_viol)**2) + np.sum(np.maximum(0, bound_viol)**2)
    
    return -t + penalty

def _epigraph_grad(x, n):
    """Gradient of epigraph objective."""
    centers = x[:2*n].reshape(n, 2)
    t = x[2*n]
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    dist_x = 0.5 - np.abs(centers[:, 0])
    dist_y = 0.5 - np.abs(centers[:, 1])
    bound_vals = np.minimum(dist_x, dist_y)
    
    pair_viol = 2.0 * t - dists
    bound_viol = t - bound_vals
    
    # Gradient of penalty w.r.t. centers
    grad_centers = np.zeros_like(centers)
    
    # Pairwise: d/d(centers[i]) sum_j max(0, 2t - dist_ij)^2
    # = sum_j 2*max(0, 2t-dist_ij) * (-1) * d(dist_ij)/d(centers[i])
    # d(dist_ij)/d(centers[i]) = (centers[i] - centers[j]) / dist_ij
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v = pair_viol[i, j]
            if v > 0:
                d = dists[i, j]
                if d > 1e-12:
                    grad_centers[i] -= 2.0 * v * (centers[i] - centers[j]) / d
    
    # Boundary: d/d(centers[i]) max(0, t - b_i)^2
    for i in range(n):
        v = bound_viol[i]
        if v > 0:
            dx = 0.5 - np.abs(centers[i, 0])
            dy = 0.5 - np.abs(centers[i, 1])
            if dx <= dy:
                gx = -np.sign(centers[i, 0]) if centers[i, 0] != 0 else 1.0
                gy = 0.0
            else:
                gx = 0.0
                gy = -np.sign(centers[i, 1]) if centers[i, 1] != 0 else 1.0
            grad_centers[i, 0] += 2.0 * v * gx
            grad_centers[i, 1] += 2.0 * v * gy
    
    # Gradient w.r.t. t: -1 + sum of penalties
    grad_t = -1.0
    pair_viol_pos = np.maximum(0, pair_viol)
    bound_viol_pos = np.maximum(0, bound_viol)
    grad_t += 2.0 * np.sum(2.0 * t - dists) * 0  # Actually: d/dt sum(max(0, 2t-d)^2) = sum(2*max(0,2t-d)*2)
    # Let me recompute: d/dt [max(0, 2t - d)^2] = 2*max(0, 2t-d) * 2 = 4*max(0, 2t-d)
    grad_t = -1.0 + 4.0 * np.sum(pair_viol_pos) + 2.0 * np.sum(bound_viol_pos)
    
    grad = np.concatenate([grad_centers.ravel(), [grad_t]])
    return grad

def _soft_min_obj(x, n, beta):
    """Soft-min objective for optimization. x is flattened centers."""
    centers = x.reshape(n, 2)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    dist_x = 0.5 - np.abs(centers[:, 0])
    dist_y = 0.5 - np.abs(centers[:, 1])
    bound_vals = np.minimum(dist_x, dist_y)
    
    all_vals = np.concatenate([pair_vals[~np.eye(n, dtype=bool)], bound_vals])
    
    m = np.min(all_vals)
    exp_terms = np.exp(-beta * (all_vals - m))
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = m - log_sum / beta
    
    return -soft_min_val

def _smart_init(n):
    """Create a smart initial configuration."""
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid_49 = np.array([(x, y) for y in ticks for x in ticks])
    
    if n == 49:
        return grid_49
    
    # Find best spot for the 50th point
    sample_res = 200
    xs = np.linspace(-0.5, 0.5, sample_res)
    ys = np.linspace(-0.5, 0.5, sample_res)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    
    diff = pts[:, np.newaxis, :] - grid_49[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    min_dists = np.min(dists, axis=1)
    
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    d_b = np.minimum(d_x, d_y)
    
    clearance = np.minimum(min_dists, d_b)
    idx = np.argmax(clearance)
    new_pt = pts[idx]
    
    centers = np.vstack([grid_49, new_pt])
    
    np.random.seed(42)
    centers = centers + np.random.normal(0, 0.003, size=centers.shape)
    
    return centers[:n]

def _sa_optimize(centers, n, t_start, t_end, n_iter, seed=0):
    """Simulated annealing for circle packing."""
    rng = np.random.RandomState(seed)
    centers = centers.copy()
    current_r = _compute_radius(centers)
    best_centers = centers.copy()
    best_r = current_r
    
    # Precompute pair indices
    pair_indices = []
    for i in range(n):
        for j in range(i+1, n):
            pair_indices.append((i, j))
    pair_indices = np.array(pair_indices)
    
    for it in range(n_iter):
        # Annealing schedule
        alpha = it / max(n_iter - 1, 1)
        temperature = t_start * (t_end / t_start) ** alpha
        step_size = 0.02 * (1.0 - alpha * 0.5)
        
        # Random perturbation
        new_centers = centers + rng.normal(0, step_size, size=centers.shape)
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_r = _compute_radius(new_centers)
        
        # Accept if better or with probability exp((new_r - current_r) / temperature)
        if new_r > current_r:
            centers = new_centers
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_centers = new_centers.copy()
        else:
            delta = new_r - current_r
            if temperature > 1e-10:
                prob = np.exp(delta / temperature)
                if rng.random() < prob:
                    centers = new_centers
                    current_r = new_r
    
    return best_centers, best_r

def _nelder_mead_polish(centers, n, max_iter=200, tol=1e-14):
    """Nelder-Mead polish for final precision."""
    x0 = centers.flatten()
    
    def obj(x):
        c = x.reshape(n, 2)
        r = _compute_radius(c)
        return -r
    
    result = minimize(
        obj,
        x0,
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True}
    )
    
    refined_centers = result.x.reshape(n, 2)
    refined_r = _compute_radius(refined_centers)
    
    return refined_centers, refined_r

def _l_bfgs_polish(centers, n, beta, max_iter=200):
    """L-BFGS-B polish using soft-min."""
    x0 = centers.flatten()
    
    result = minimize(
        _soft_min_obj,
        x0,
        args=(n, beta),
        method='L-BFGS-B',
        bounds=[(-0.499, 0.499)] * (2 * n),
        options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12}
    )
    
    refined_centers = result.x.reshape(n, 2)
    refined_r = _compute_radius(refined_centers)
    
    return refined_centers, refined_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_centers = None
    best_r = 0.0
    
    # Phase 1: Multiple SA runs with different seeds and initializations
    inits = [
        _smart_init(N),
        _smart_init(N) + np.random.RandomState(1).normal(0, 0.03, size=(N, 2)),
        _smart_init(N) + np.random.RandomState(2).normal(0, 0.05, size=(N, 2)),
    ]
    
    sa_configs = [
        (0.05, 0.001, 800, 0),
        (0.04, 0.001, 1000, 1),
        (0.03, 0.0005, 1200, 2),
    ]
    
    for init_idx, init_centers in enumerate(inits):
        if time.time() > deadline - 40.0:
            break
        
        init_centers = np.clip(init_centers, -0.49, 0.49)
        
        # Try multiple SA configurations
        for sa_idx in range(min(3, len(sa_configs))):
            if time.time() > deadline - 35.0:
                break
            
            t_start, t_end, n_iter, seed = sa_configs[sa_idx]
            
            sa_centers, sa_r = _sa_optimize(init_centers, N, t_start, t_end, n_iter, seed)
            
            if sa_r > best_r:
                best_r = sa_r
                best_centers = sa_centers.copy()
            
            # Quick L-BFGS polish after SA
            if time.time() < deadline - 20.0:
                polished_centers, polished_r = _l_bfgs_polish(sa_centers, N, beta=100, max_iter=100)
                if polished_r > best_r:
                    best_r = polished_r
                    best_centers = polished_centers.copy()
    
    # Phase 2: Final Nelder-Mead polish on best
    if best_centers is not None and time.time() < deadline - 10.0:
        nm_centers, nm_r = _nelder_mead_polish(best_centers, N, max_iter=300, tol=1e-14)
        if nm_r > best_r:
            best_r = nm_r
            best_centers = nm_centers.copy()
    
    # Phase 3: Final L-BFGS with high beta
    if best_centers is not None and time.time() < deadline - 5.0:
        lbfgs_centers, lbfgs_r = _l_bfgs_polish(best_centers, N, beta=500, max_iter=100)
        if lbfgs_r > best_r:
            best_r = lbfgs_r
            best_centers = lbfgs_centers.copy()
    
    # Ensure validity
    if best_centers is not None:
        eps = 1e-10
        best_centers = np.clip(best_centers, -0.5 + eps, 0.5 - eps)
        # Final check
        final_r = _compute_radius(best_centers)
        if final_r <= 0:
            best_centers = _smart_init(N)
            best_centers = np.clip(best_centers, -0.5 + 1e-9, 0.5 - 1e-9)
    else:
        best_centers = _smart_init(N)
        best_centers = np.clip(best_centers, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_centers

# EVOLVE-BLOCK-END
