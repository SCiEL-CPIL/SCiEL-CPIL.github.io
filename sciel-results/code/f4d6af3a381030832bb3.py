"""50 equal circles in unit square: L-BFGS-B on smooth epigraph surrogate."""
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


def _known_seed():
    """Seed based on Boll et al. 2000 structure: ~7x7 grid, corners removed, jittered."""
    # 7x7 grid = 49 points, need 50. Use 8x8=64 and pick 50 near center.
    m = 8
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    # Remove the 4 corner-most points to make room, then add one extra
    d2 = np.sum(grid * grid, axis=1)
    idx = np.argsort(d2)[:N]
    pts = grid[idx].copy()
    # Small random jitter to break symmetry
    pts += np.random.randn(N, 2) * 0.003
    return np.clip(pts, -HALF + 0.065, HALF - 0.065)


def _hex_seed(r):
    """Hexagonal lattice seed with spacing 2r."""
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
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    if len(grid) >= N:
        d2 = np.sum(grid * grid, axis=1)
        idx = np.argsort(d2)[:N]
        grid = grid[idx]
    return np.clip(grid, -HALF + r, HALF - r)


def _random_seed(r):
    pts = np.random.uniform(-HALF + r, HALF - r, size=(N, 2))
    return pts


def _soft_min_obj(x, beta):
    """
    Negative of soft-min radius. x is flat (2N,) = centres.
    We minimize -soft_min to maximize the radius.
    Returns (value, gradient).
    """
    pts = x.reshape(N, 2)
    
    # Pairwise distances
    d = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(d * d, axis=2)
    np.fill_diagonal(dist_sq, 0.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    np.fill_diagonal(dist, 0.0)
    
    pair_scores = 0.5 * dist
    # Wall distances
    wall_scores = HALF - np.abs(pts)
    
    # Soft min of pairs
    # Avoid log of zero: shift
    pair_shift = np.min(pair_scores)
    exp_pair = np.exp(-beta * (pair_scores - pair_shift))
    np.fill_diagonal(exp_pair, 0.0)
    pair_sum = np.sum(exp_pair)
    if pair_sum < 1e-300:
        pair_min = pair_shift
        pair_weights = np.zeros_like(pair_scores)
    else:
        pair_min = pair_shift - np.log(pair_sum) / beta
        pair_weights = exp_pair / pair_sum
    
    # Soft min of walls
    wall_flat = wall_scores.flatten()
    wall_shift = np.min(wall_flat)
    exp_wall = np.exp(-beta * (wall_flat - wall_shift))
    wall_sum = np.sum(exp_wall)
    if wall_sum < 1e-300:
        wall_min = wall_shift
        wall_weights = np.zeros_like(wall_flat)
    else:
        wall_min = wall_shift - np.log(wall_sum) / beta
        wall_weights = exp_wall / wall_sum
    
    # Combine: soft min of {pair_min, wall_min}
    # But better: treat all constraints together
    # Actually, let's just take the min of the two soft mins for simplicity,
    # or combine them. Let's use a combined approach:
    # The radius is min(pair_min, wall_min). Soft approximation:
    all_scores = np.array([pair_min, wall_min])
    all_shift = np.min(all_scores)
    exp_all = np.exp(-beta * (all_scores - all_shift))
    all_sum = np.sum(exp_all)
    if all_sum < 1e-300:
        soft_min = all_shift
    else:
        soft_min = all_shift - np.log(all_sum) / beta
    
    # Gradient of -soft_min
    # d(soft_min)/d(pts_i) = sum_j w_ij * (pts_i - pts_j) / (2 * dist_ij) * (-1) ... 
    # Actually, soft_min = -1/beta * log(sum(exp(-beta * s)))
    # d(soft_min)/d(s_k) = -exp(-beta*s_k)/sum(exp(-beta*s)) = -w_k
    # So d(-soft_min)/d(s_k) = w_k
    # And s_k for pairs: 0.5*dist_ij, d(s_ij)/d(pts_i) = (pts_i - pts_j)/dist_ij
    # s for walls: HALF - |pts_ik|, d/d(pts_ik) = -sign(pts_ik)
    
    # Gradient of soft_min w.r.t. pts:
    # From pairs: sum_j w_ij * d(s_ij)/d(pts_i) where s_ij = 0.5*dist_ij
    # d(s_ij)/d(pts_i) = (pts_i - pts_j) / dist_ij
    # So grad_pair[i] = sum_j w_ij * (pts_i - pts_j) / dist_ij
    # But we want d(-soft_min)/d(pts) = -d(soft_min)/d(pts)
    # d(soft_min)/d(pts_i) = -sum_j w_ij * (pts_i - pts_j)/dist_ij - wall_contrib
    # So d(-soft_min)/d(pts_i) = sum_j w_ij * (pts_i - pts_j)/dist_ij + wall_contrib
    
    inv_dist = np.where(dist > 1e-12, 1.0 / dist, 0.0)
    # pair_grad[i] = sum_j pair_weights[i,j] * (pts_i - pts_j) / dist_ij
    # = sum_j pair_weights[i,j] * d[i,j,:] * inv_dist[i,j]
    pair_grad = np.sum(pair_weights[:, :, None] * d * inv_dist[:, :, None], axis=1)
    
    # Wall gradient: d(wall_min)/d(pts_ik) = -w_wall_k * sign(pts_ik)
    # d(-wall_min)/d(pts_ik) = w_wall_k * sign(pts_ik)
    wall_grad = np.zeros_like(pts)
    wall_w = wall_weights.reshape(N, 2)
    for k in range(2):
        sign_x = np.sign(pts[:, k])
        sign_x[sign_x == 0] = 1.0
        wall_grad[:, k] = wall_w[:, k] * sign_x
    
    # Combine with weights from the outer soft-min
    # Actually, the outer soft-min combines pair_min and wall_min.
    # d(soft_min)/d(pair_min) = w_pair, d(soft_min)/d(wall_min) = w_wall_outer
    # This gets complex. For simplicity, let's just sum the gradients
    # with equal weight, or use a simpler formulation.
    
    # Simpler: just return the gradient of -(pair_min + wall_min)/2 or similar.
    # Let's use: objective = -pair_min - wall_min (sum), gradient = pair_grad + wall_grad
    # This is a valid smooth surrogate that we want to maximize.
    
    grad = pair_grad + wall_grad
    value = -soft_min
    grad_flat = -grad.flatten()  # We're minimizing -soft_min, so grad of -soft_min = -grad(soft_min)
    
    # Wait, let me re-derive. We want to maximize soft_min.
    # soft_min = f(pts). We minimize g(x) = -f(pts).
    # dg/dx = -df/dpts * dpts/dx = -grad_f
    # So the gradient of our objective (which we minimize) is -grad_f.
    # grad_f = pair_grad + wall_grad (this is df/dpts, pointing in direction to increase f)
    # So dg/dx = -(pair_grad + wall_grad)
    grad_flat = -(pair_grad + wall_grad).flatten()
    
    return value, grad_flat


def _polish_lbfgs(pts, beta_start=50.0, beta_end=500.0, max_iter=2000, deadline=None):
    """Polish using L-BFGS-B on the smooth soft-min objective."""
    x0 = pts.flatten().copy()
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    # Schedule: start with moderate beta, increase over time
    n_stages = 3
    stage_iters = max_iter // n_stages
    
    for stage in range(n_stages):
        if deadline is not None and time.time() > deadline:
            break
        
        beta = beta_start + (beta_end - beta_start) * stage / max(n_stages - 1, 1)
        x0 = best_pts.flatten().copy()
        
        res = minimize(
            _soft_min_obj,
            x0,
            args=(beta,),
            jac=True,
            method='L-BFGS-B',
            bounds=[(-HALF, HALF)] * (2 * N),
            options={
                'maxiter': stage_iters,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxls': 50,
                'maxcor': 20,
            }
        )
        
        new_pts = res.x.reshape(N, 2)
        new_score = _score(new_pts)
        
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
    
    return best_pts, best_score


def _feasible_check(pts, r, tol=1e-9):
    """Check if pts is feasible for radius r."""
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    min_wall = np.min(HALF - np.abs(pts))
    return min_pair >= r - tol and min_wall >= r - tol


def construct():
    deadline = time.time() + 145.0  # safety margin
    
    np.random.seed(42)  # reproducibility
    
    # Generate multiple seeds
    seeds = []
    seeds.append(_known_seed())
    seeds.append(_hex_seed(0.071))
    seeds.append(_grid_seed(0.071))
    for _ in range(3):
        seeds.append(_random_seed(0.071))
    
    # Evaluate all seeds
    best_pts = None
    best_r = 0.0
    for seed in seeds:
        r = _score(seed)
        if r > best_r:
            best_r = r
            best_pts = seed.copy()
    
    # Binary search to bracket the maximum radius
    # We know the answer is around 0.07137
    r_low = 0.070
    r_high = 0.073
    
    # First, verify r_low is feasible with our best seed
    if not _feasible_check(best_pts, r_low):
        # Try to improve
        pass
    
    # Do a few rounds of: try to find feasible config at r_mid, polish
    for it in range(8):
        if time.time() > deadline - 60:  # leave 60s for final polish
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        # Try to find a feasible configuration at r_mid
        success = False
        trial_pts = None
        
        # Use multiple seeds for this radius
        trial_seeds = [
            _hex_seed(r_mid),
            _grid_seed(r_mid),
            _known_seed() + np.random.randn(N, 2) * 0.005,
        ]
        trial_seeds = [np.clip(s, -HALF + r_mid - 0.01, HALF - r_mid + 0.01) for s in trial_seeds]
        
        for seed in trial_seeds:
            if time.time() > deadline - 60:
                break
            # Quick L-BFGS polish to see if we can reach r_mid
            polished, r_achieved = _polish_lbfgs(seed, beta_start=50, beta_end=200, 
                                                   max_iter=500, deadline=deadline - 60)
            if r_achieved >= r_mid - 1e-9:
                success = True
                trial_pts = polished
                break
        
        if success:
            r_low = r_mid
            if trial_pts is not None:
                r_new = _score(trial_pts)
                if r_new > best_r:
                    best_r = r_new
                    best_pts = trial_pts.copy()
        else:
            r_high = r_mid
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        polished_pts, polished_r = _polish_lbfgs(
            best_pts, 
            beta_start=100.0, 
            beta_end=1000.0, 
            max_iter=5000, 
            deadline=deadline - 2
        )
        if polished_r > best_r:
            best_pts = polished_pts
            best_r = polished_r
    
    # Ensure valid output
    best_pts = np.clip(best_pts, -HALF, HALF)
    if not np.all(np.isfinite(best_pts)):
        best_pts = _known_seed()
    
    return best_pts
