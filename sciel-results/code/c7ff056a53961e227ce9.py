"""Binary search on radius with projected gradient descent for 50 circles in unit square."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
HALF = 0.5


def _score(c):
    """Largest common radius for centres c (N,2)."""
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    min_bound = np.min(HALF - np.abs(c))
    return min(min_pair, min_bound)


def _feasible(pts, r, max_iter=2000, deadline=None):
    """
    Try to find a configuration of N circles of radius r inside the square.
    Uses projected gradient descent to push overlapping circles apart and
    keep them within bounds. Returns (success, pts).
    """
    pts = pts.astype(np.float64).copy()
    n = len(pts)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute all pairwise distances and forces
        diff = pts[:, None, :] - pts[None, :, :]  # (n, n, 2)
        dist = np.sqrt(np.sum(diff * diff, axis=2))  # (n, n)
        np.fill_diagonal(dist, np.inf)
        
        # Compute forces for overlapping pairs
        overlap = (dist < 2 * r) & (dist > 1e-12)
        force_mag = np.where(overlap, (2 * r - dist), 0.0)
        inv_dist = np.where(dist > 1e-12, 1.0 / dist, 0.0)
        force = diff * (force_mag * inv_dist)[:, :, None]
        grad = np.sum(force, axis=1)  # (n, 2)
        
        # Wall constraints: push away from walls
        wall_force = np.zeros_like(pts)
        for k in range(2):
            too_right = pts[:, k] > HALF - r
            too_left = pts[:, k] < -HALF + r
            wall_force[too_right, k] = -(pts[too_right, k] - (HALF - r))
            wall_force[too_left, k] = (pts[too_left, k] - (-HALF + r))
        
        grad += wall_force
        
        # Adaptive step size
        step = 0.05
        new_pts = pts + step * grad
        new_pts = np.clip(new_pts, -HALF + r, HALF - r)
        
        # Check if all constraints are satisfied
        min_pair = np.min(dist)
        min_wall = np.min(HALF - np.abs(new_pts))
        
        if min_pair >= 2 * r - 1e-10 and min_wall >= r - 1e-10:
            return True, new_pts
        
        pts = new_pts
        
        # Early termination if not making progress
        if it > 100:
            cur_min = min(np.min(np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2))),
                          np.min(HALF - np.abs(pts)))
            if cur_min < 2 * r - 1e-8:
                if it > 500:
                    break
    
    # Final check
    dist = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    min_wall = np.min(HALF - np.abs(pts))
    success = (min_pair >= 2 * r - 1e-10) and (min_wall >= r - 1e-10)
    return success, pts


def _hex_seed(r):
    """Generate a hexagonal lattice seed with spacing 2r."""
    dx = 2.0 * r
    dy = np.sqrt(3.0) * r
    xs, ys = [], []
    row = 0
    y = -HALF + r
    while y <= HALF - r + 1e-9:
        x = -HALF + r + (0.5 * dx if (row % 2) else 0.0)
        while x <= HALF - r + 1e-9:
            xs.append(x)
            ys.append(y)
            x += dx
        y += dy
        row += 1
    pts = np.array(list(zip(xs, ys)))
    if len(pts) >= N:
        d2 = np.sum(pts * pts, axis=1)
        idx = np.argsort(d2)[:N]
        pts = pts[idx]
    else:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, yy) for yy in ticks for x in ticks])
        pts = np.vstack([pts, grid[:N - len(pts)]])
    return np.clip(pts, -HALF + r, HALF - r)


def _grid_seed(r):
    """Generate a grid seed."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    if len(grid) >= N:
        d2 = np.sum(grid * grid, axis=1)
        idx = np.argsort(d2)[:N]
        grid = grid[idx]
    return np.clip(grid, -HALF + r, HALF - r)


def _soft_min_score(c, beta=50.0):
    """Smooth approximation of the minimum radius using log-sum-exp."""
    c = np.asarray(c, dtype=np.float64)
    d = c[:, None, :] - c[None, :, :]
    dist_sq = np.sum(d * d, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    # Pairwise distances / 2
    pair_scores = 0.5 * np.sqrt(dist_sq)
    # Wall distances
    wall_scores = HALF - np.abs(c)
    
    # Soft min of pair scores
    pair_min = -np.log(np.sum(np.exp(-beta * pair_scores))) / beta
    # Soft min of wall scores  
    wall_min = -np.log(np.sum(np.exp(-beta * wall_scores))) / beta
    
    # Soft min of both
    all_scores = np.concatenate([pair_min, wall_min])
    soft_min = -np.log(np.sum(np.exp(-beta * all_scores))) / beta
    return soft_min


def _smooth_obj_and_grad(x_flat, beta=100.0):
    """
    Objective: -softmin(distances)
    Gradient: -d(softmin)/dx
    """
    pts = x_flat.reshape(N, 2)
    
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_dists = 0.5 * dist
    
    # Wall distances (4 per point)
    wall_dists = np.concatenate([
        (HALF - pts[:, 0]).flatten(),
        (HALF + pts[:, 0]).flatten(),
        (HALF - pts[:, 1]).flatten(),
        (HALF + pts[:, 1]).flatten()
    ])
    
    all_dists = np.concatenate([pair_dists.flatten(), wall_dists])
    
    # Soft min
    exp_dists = np.exp(-beta * all_dists)
    sum_exp = np.sum(exp_dists)
    soft_min = -np.log(sum_exp) / beta
    obj = -soft_min
    
    # Gradient of soft_min w.r.t. all_dists
    weights = exp_dists / sum_exp
    
    # Gradient w.r.t. pts
    grad_pts = np.zeros((N, 2))
    
    # Pairwise gradient
    exp_pair = np.exp(-beta * pair_dists)
    exp_pair[np.diag_indices_from(exp_pair)] = 0
    sum_exp_pair = np.sum(exp_pair)
    sum_exp_wall = np.sum(np.exp(-beta * wall_dists))
    total_sum = sum_exp_pair + sum_exp_wall
    
    # Weights for pairs in the total soft-min
    w_pair = exp_pair / total_sum
    
    # Gradient from pairs
    inv_dist = np.where(dist > 1e-12, 1.0 / dist, 0.0)
    pair_grad = np.sum(w_pair[:, :, None] * diff * inv_dist[:, :, None], axis=1)
    
    # Gradient from walls
    w_wall = np.exp(-beta * wall_dists) / total_sum
    
    # Wall gradients
    wall_grad = np.zeros((N, 2))
    wall_grad[:, 0] -= w_wall[0:N]       # 0.5 - x
    wall_grad[:, 0] += w_wall[N:2*N]     # 0.5 + x
    wall_grad[:, 1] -= w_wall[2*N:3*N]   # 0.5 - y
    wall_grad[:, 1] += w_wall[3*N:4*N]   # 0.5 + y
    
    total_grad = pair_grad + wall_grad
    
    # Objective is -soft_min, so gradient is - (gradient of soft_min)
    grad_flat = -total_grad.flatten()
    
    return obj, grad_flat


def _polish(pts, max_iter=5000, deadline=None):
    """
    High-precision polish using L-BFGS-B on smooth surrogate.
    """
    pts = pts.astype(np.float64).copy()
    best_score = _score(pts)
    best_pts = pts.copy()
    
    beta = 50.0
    
    # Use L-BFGS-B
    res = minimize(
        _smooth_obj_and_grad, 
        pts.flatten(), 
        args=(beta,),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 200, 'gtol': 1e-8}
    )
    
    new_pts = res.x.reshape(N, 2)
    new_pts = np.clip(new_pts, -HALF, HALF)
    
    new_score = _score(new_pts)
    
    if new_score > best_score:
        best_pts = new_pts
        best_score = new_score
    
    return best_pts, best_score


def construct():
    deadline = time.time() + 145.0  # safety margin before 150s kill
    
    # Start with a reasonable radius estimate
    r_low = 0.06
    r_high = 0.08
    
    best_pts = _hex_seed(0.071).copy()
    best_r = _score(best_pts)
    
    # Binary search on radius
    for it in range(30):
        if time.time() > deadline:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        success = False
        trial_pts = None
        
        seeds = [_hex_seed(r_mid), _grid_seed(r_mid)]
        for _ in range(2):
            seed = _hex_seed(r_mid) + np.random.randn(N, 2) * 0.01
            seed = np.clip(seed, -HALF + r_mid, HALF + r_mid)
            seeds.append(seed)
        
        for seed in seeds:
            if time.time() > deadline:
                break
            ok, pts = _feasible(seed, r_mid, max_iter=1000, deadline=deadline)
            if ok:
                success = True
                trial_pts = pts
                break
        
        if success:
            r_low = r_mid
            best_pts = trial_pts.copy()
            best_r = _score(best_pts)
        else:
            r_high = r_mid
    
    # Final refinement with smooth polish
    if time.time() < deadline - 5:
        # Try a few perturbations
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            perturbed = best_pts + np.random.randn(N, 2) * 0.005
            perturbed = np.clip(perturbed, -HALF, HALF)
            polished_pts, polished_r = _polish(perturbed, deadline=deadline - 2)
            if polished_r > best_r:
                best_pts = polished_pts
                best_r = polished_r
    
    return best_pts
