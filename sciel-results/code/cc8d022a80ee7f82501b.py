"""Pack 200 equal circles in the unit square [-1/2, 1/2]^2 to maximize common radius."""
import time
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

N = 200
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill


def _remaining_time():
    return TIME_LIMIT - (time.time() - START_TIME)


def _compute_radius(centers):
    """Compute the maximum radius for given centers."""
    if len(centers) == 0:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall distances
    dx = 0.5 - np.abs(centers[:, 0])
    dy = 0.5 - np.abs(centers[:, 1])
    wall_min = min(np.min(dx), np.min(dy))
    if wall_min < 0:
        return 0.0
    # Pairwise distances
    dists = pdist(centers)
    pair_min = np.min(dists) / 2.0
    return min(wall_min, pair_min)


def _hexagonal_grid(n):
    """Generate a hexagonal grid of n points scaled to fit in [-0.5, 0.5]^2."""
    m = int(np.ceil(np.sqrt(n)))
    ticks_x = (np.arange(m) + 0.5) / m - 0.5
    ticks_y = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks_y for x in ticks_x])
    
    if len(grid) > n:
        grid = grid[:n]
    elif len(grid) < n:
        extra = np.random.randn(n - len(grid), 2) * 0.1
        extra[:, 0] = np.clip(extra[:, 0], -0.4, 0.4)
        extra[:, 1] = np.clip(extra[:, 1], -0.4, 0.4)
        grid = np.vstack([grid, extra])
    
    return grid


def _soft_min_objective_and_grad(x, t=0.01):
    """
    Compute the soft-min objective and its gradient.
    x is a flat array of length 2*N.
    """
    pts = x.reshape(-1, 2)
    
    # Wall distances
    dx = 0.5 - np.abs(pts[:, 0])
    dy = 0.5 - np.abs(pts[:, 1])
    wall_dists = np.concatenate([dx, dy])
    
    # Pairwise distances / 2
    dists = pdist(pts)
    pair_dists = dists / 2.0
    
    all_dists = np.concatenate([wall_dists, pair_dists])
    
    # Soft min
    m = np.max(all_dists)
    exp_vals = np.exp(-(all_dists - m) / t)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    soft_min_val = m - t * np.log(sum_exp)
    
    # Gradient of soft_min with respect to pts
    grad = np.zeros_like(pts)
    
    # Wall gradients
    gx_wall = -np.sign(pts[:, 0])
    gy_wall = -np.sign(pts[:, 1])
    
    grad[:, 0] += weights[:N] * gx_wall
    grad[:, 1] += weights[N:2*N] * gy_wall
    
    # Pairwise gradients
    i_idx, j_idx = np.triu_indices(N, k=1)
    num_pairs = len(i_idx)
    
    weights_pair = weights[2*N:2*N + num_pairs]
    
    diffs = pts[i_idx] - pts[j_idx]
    safe_dists = np.where(dists > 1e-10, dists, 1.0)
    
    terms = weights_pair[:, np.newaxis] * diffs / (2.0 * safe_dists[:, np.newaxis])
    
    np.add.at(grad, i_idx, terms)
    np.add.at(grad, j_idx, -terms)
    
    obj_val = -soft_min_val
    grad_val = -grad.flatten()
    
    return obj_val, grad_val


def _optimize_l_bfgs(x0, max_iter=300, t=0.01):
    """Optimize using L-BFGS-B on the soft-min objective."""
    def objective_and_grad(x):
        val, grad = _soft_min_objective_and_grad(x, t)
        return val, grad
    
    res = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B',
                   options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 10000})
    return res.x.reshape(-1, 2)


def _refine(pts, max_iter=200, t=0.01):
    """Refine a configuration using L-BFGS-B."""
    x0 = pts.flatten()
    pts_opt = _optimize_l_bfgs(x0, max_iter=max_iter, t=t)
    return pts_opt


def _repulsion_energy(x, r_target=0.0366):
    """Compute repulsion energy for given points and target radius."""
    pts = x.reshape(-1, 2)
    energy = 0.0
    
    # Wall repulsion
    for i in range(N):
        for d in [0, 1]:
            dist = 0.5 - abs(pts[i, d])
            if dist < r_target:
                energy += (r_target - dist) ** 2
    
    # Pair repulsion
    dists = pdist(pts)
    for k in range(len(dists)):
        if dists[k] < 2 * r_target:
            energy += (2 * r_target - dists[k]) ** 2
    
    return energy


def _repulsion_grad(x, r_target=0.0366):
    """Compute gradient of repulsion energy."""
    pts = x.reshape(-1, 2)
    grad = np.zeros_like(pts)
    
    # Wall gradient
    for i in range(N):
        for d in [0, 1]:
            dist = 0.5 - abs(pts[i, d])
            if dist < r_target:
                sign = -np.sign(pts[i, d])
                grad[i, d] += 2 * (r_target - dist) * sign
    
    # Pair gradient
    i_idx, j_idx = np.triu_indices(N, k=1)
    dists = pdist(pts)
    diffs = pts[i_idx] - pts[j_idx]
    safe_dists = np.where(dists > 1e-10, dists, 1.0)
    
    mask = dists < 2 * r_target
    for k in range(len(dists)):
        if mask[k]:
            coeff = 2 * (2 * r_target - dists[k]) / safe_dists[k]
            grad[i_idx[k]] += coeff * diffs[k]
            grad[j_idx[k]] -= coeff * diffs[k]
    
    return grad.flatten()


def _optimize_repulsion(x0, r_target=0.0366, max_iter=500):
    """Optimize using L-BFGS-B on repulsion energy."""
    def objective_and_grad(x):
        return _repulsion_energy(x, r_target), _repulsion_grad(x, r_target)
    
    res = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B',
                   options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 20000})
    return res.x.reshape(-1, 2)


def construct():
    best_pts = None
    best_r = 0.0
    
    np.random.seed(42)
    
    # Initial guess: hexagonal-like grid
    pts0 = _hexagonal_grid(N)
    
    # Optimize the initial guess with soft-min
    pts_opt = _refine(pts0, max_iter=300, t=0.01)
    r = _compute_radius(pts_opt)
    if r > best_r:
        best_r = r
        best_pts = pts_opt.copy()
    
    # Try repulsion-based optimization from the best point
    if best_pts is not None and _remaining_time() > 10.0:
        pts_rep = _optimize_repulsion(best_pts.flatten(), r_target=0.0366, max_iter=500)
        r = _compute_radius(pts_rep)
        if r > best_r:
            best_r = r
            best_pts = pts_rep.copy()
    
    # Random restarts with different initial perturbations
    max_restarts = 50
    for i in range(max_restarts):
        if _remaining_time() < 8.0:
            break
        
        # Generate a random initial guess based on the grid
        noise_scale = 0.003 + 0.008 * (i % 5)
        pts_rand = _hexagonal_grid(N) + np.random.randn(N, 2) * noise_scale
        pts_rand[:, 0] = np.clip(pts_rand[:, 0], -0.49, 0.49)
        pts_rand[:, 1] = np.clip(pts_rand[:, 1], -0.49, 0.49)
        
        # Optimize with soft-min
        pts_opt = _refine(pts_rand, max_iter=150, t=0.01)
        r = _compute_radius(pts_opt)
        if r > best_r:
            best_r = r
            best_pts = pts_opt.copy()
        
        # Also try repulsion from this point if time permits
        if _remaining_time() > 15.0 and r > best_r - 0.001:
            pts_rep = _optimize_repulsion(pts_opt.flatten(), r_target=0.0366, max_iter=300)
            r_rep = _compute_radius(pts_rep)
            if r_rep > best_r:
                best_r = r_rep
                best_pts = pts_rep.copy()
    
    # Final polish on the best solution with multiple temperature values
    if best_pts is not None and _remaining_time() > 5.0:
        for t in [0.005, 0.01, 0.02]:
            if _remaining_time() < 3.0:
                break
            pts_polish = _refine(best_pts, max_iter=200, t=t)
            r = _compute_radius(pts_polish)
            if r > best_r:
                best_r = r
                best_pts = pts_polish.copy()
        
        # One more repulsion polish
        if _remaining_time() > 5.0:
            pts_rep = _optimize_repulsion(best_pts.flatten(), r_target=best_r, max_iter=500)
            r = _compute_radius(pts_rep)
            if r > best_r:
                best_r = r
                best_pts = pts_rep.copy()
    
    return best_pts
