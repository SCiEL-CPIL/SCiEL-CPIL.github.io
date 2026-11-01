import numpy as np
from scipy.optimize import minimize
import time

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


def _feasible(pts, r, max_iter=3000, deadline=None):
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
        dist_new = np.sqrt(np.sum((new_pts[:, None, :] - new_pts[None, :, :]) ** 2, axis=2))
        np.fill_diagonal(dist_new, np.inf)
        min_pair = np.min(dist_new)
        min_wall = np.min(HALF - np.abs(new_pts))
        
        if min_pair >= 2 * r - 1e-10 and min_wall >= r - 1e-10:
            return True, new_pts
        
        pts = new_pts
        
        # Early termination if not making progress
        if it > 100:
            cur_min = min(np.min(dist_new), min_wall)
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


def _soft_min_obj(pts, beta=50.0):
    """
    Smooth surrogate objective: minimize -soft_min(radius).
    Returns (value, gradient).
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Pair scores: dist / 2
    pair_scores = 0.5 * dist
    # Wall scores
    wall_scores = HALF - np.abs(pts)
    
    # Soft min via log-sum-exp
    # For numerical stability, subtract max
    def soft_min(vals, beta):
        vals = np.asarray(vals)
        max_val = np.max(vals)
        exps = np.exp(-beta * (vals - max_val))
        return max_val + np.log(np.sum(exps)) / beta
    
    def soft_min_grad(pts, beta):
        """Gradient of soft_min w.r.t. pts."""
        d = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt(np.sum(d * d, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        pair_scores = 0.5 * dist
        wall_scores = HALF - np.abs(pts)
        
        # Weights for pairs
        max_pair = np.max(pair_scores)
        exp_pair = np.exp(-beta * (pair_scores - max_pair))
        np.fill_diagonal(exp_pair, 0)
        sum_exp_pair = np.sum(exp_pair)
        pair_weights = exp_pair / sum_exp_pair
        
        # Gradient of pair soft min w.r.t. pts[i]
        # d(soft_min)/d(pts[i]) = sum_j w_ij * d(pair_score_ij)/d(pts[i])
        # d(0.5 * ||pts_i - pts_j||)/d(pts_i) = (pts_i - pts_j) / (2 * ||pts_i - pts_j||)
        inv_dist = np.where(dist > 1e-12, 1.0 / dist, 0.0)
        pair_grad = np.sum(pair_weights[:, :, None] * d * (0.5 * inv_dist)[:, :, None], axis=1)
        
        # Weights for walls
        max_wall = np.max(wall_scores)
        exp_wall = np.exp(-beta * (wall_scores - max_wall))
        sum_exp_wall = np.sum(exp_wall)
        wall_weights = exp_wall / sum_exp_wall
        
        # Gradient of wall soft min w.r.t. pts[i,k]
        # d(HALF - |x|)/dx = -sign(x)
        wall_grad = np.zeros_like(pts)
        for k in range(2):
            sign_x = np.sign(pts[:, k])
            sign_x[sign_x == 0] = 1.0
            wall_grad[:, k] = -sign_x * wall_weights[:, k]
        
        return pair_grad + wall_grad
    
    sm_pair = soft_min(pair_scores, beta)
    sm_wall = soft_min(wall_scores, beta)
    
    # Combine: soft min of the two soft mins
    # Actually, just take min of the two soft mins as a simple combination
    # Or better: concatenate all scores and take soft min
    all_scores = np.concatenate([pair_scores.ravel(), wall_scores.ravel()])
    sm_all = soft_min(all_scores, beta)
    
    grad = soft_min_grad(pts, beta)
    
    return -sm_all, -grad


def _polish(pts, max_iter=5000, deadline=None):
    """
    High-precision polish using smooth surrogate and adaptive steps.
    Maximizes the minimum distance (radius) using soft-min surrogate.
    """
    pts = pts.astype(np.float64).copy()
    n = len(pts)
    best_score = _score(pts)
    best_pts = pts.copy()
    
    beta = 30.0  # Start with moderate sharpness
    step = 0.01
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute gradient of soft-min score
        d = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt(np.sum(d * d, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        pair_scores = 0.5 * dist
        wall_scores = HALF - np.abs(pts)
        all_scores = np.concatenate([pair_scores.ravel(), wall_scores.ravel()])
        
        max_val = np.max(all_scores)
        exps = np.exp(-beta * (all_scores - max_val))
        sm = max_val + np.log(np.sum(exps)) / beta
        
        # Weights
        pair_weights = exps[:int(N*N)].reshape(N, N)
        np.fill_diagonal(pair_weights, 0)
        sum_pw = np.sum(pair_weights)
        if sum_pw > 0:
            pair_weights = pair_weights / sum_pw
        else:
            pair_weights = np.zeros((N, N))
        
        wall_weights = exps[int(N*N):]
        sum_ww = np.sum(wall_weights)
        if sum_ww > 0:
            wall_weights = wall_weights / sum_ww
        else:
            wall_weights = np.zeros(N)
        
        # Gradient
        inv_dist = np.where(dist > 1e-12, 1.0 / dist, 0.0)
        pair_grad = np.sum(pair_weights[:, :, None] * d * (0.5 * inv_dist)[:, :, None], axis=1)
        
        wall_grad = np.zeros_like(pts)
        for k in range(2):
            sign_x = np.sign(pts[:, k])
            sign_x[sign_x == 0] = 1.0
            wall_grad[:, k] = -sign_x * wall_weights[:, k]
        
        grad = pair_grad + wall_grad
        
        # Normalize gradient for stable steps
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-15:
            grad = grad / grad_norm
        
        # Adaptive step size
        new_pts = pts + step * grad
        new_pts = np.clip(new_pts, -HALF, HALF)
        
        new_score = _score(new_pts)
        
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            pts = new_pts
            step = min(step * 1.1, 0.05)
        else:
            step = step * 0.7
        
        # Increase beta for sharper approximation over time
        if it % 100 == 0 and it > 0:
            beta = min(beta * 1.5, 200.0)
        
        # Convergence check
        if it > 500 and new_score - best_score < 1e-12:
            if it > 1000:
                break
    
    return best_pts, best_score


def _lbfgs_polish(pts, deadline=None):
    """
    Polish using L-BFGS-B on the soft-min surrogate.
    """
    def obj_and_grad(x):
        if deadline is not None and time.time() > deadline:
            raise Exception("deadline")
        val, grad = _soft_min_obj(x.reshape(N, 2), beta=100.0)
        return val, grad.ravel()
    
    x0 = pts.reshape(-1)
    bounds = [(-HALF, HALF)] * (2 * N)
    
    try:
        res = minimize(obj_and_grad, x0, jac=True, method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
        if res.fun < obj_and_grad(x0)[0]:
            new_pts = res.x.reshape(N, 2)
            new_score = _score(new_pts)
            cur_score = _score(pts)
            if new_score > cur_score:
                return new_pts
    except Exception:
        pass
    
    return pts


def construct():
    start_time = time.time()
    max_time = 140.0  # Safety margin before 150s kill
    
    best_X = None
    best_val = -np.inf
    
    # Start with a reasonable radius estimate
    # Boll et al. found r = 0.071377103865
    # Let's try to find something close to that
    r_target = 0.071377103865
    
    # Try to find a feasible configuration for r_target
    # Use hex seed
    seed_pts = _hex_seed(r_target)
    success, pts = _feasible(seed_pts, r_target, max_iter=3000, deadline=start_time + max_time - 20)
    
    if success:
        best_X = pts
        best_val = _score(pts)
    else:
        # If not feasible, try a slightly smaller radius
        # Binary search for the largest feasible radius
        lo, hi = 0.06, 0.072
        for _ in range(15):
            mid = (lo + hi) / 2
            seed_pts = _hex_seed(mid)
            success, pts = _feasible(seed_pts, mid, max_iter=2000, deadline=start_time + max_time - 20)
            if success:
                lo = mid
                best_X = pts
                best_val = _score(pts)
            else:
                hi = mid
            if time.time() - start_time > max_time - 20:
                break
        
        if best_X is None:
            # Fallback to grid
            best_X = _grid_seed(0.06)
            best_val = _score(best_X)
    
    # Polish the best configuration found
    if time.time() - start_time < max_time - 10:
        polished_pts, polished_score = _polish(best_X, max_iter=5000, deadline=start_time + max_time - 10)
        if polished_score > best_val:
            best_X = polished_pts
            best_val = polished_score
    
    # Final L-BFGS polish
    if time.time() - start_time < max_time - 5:
        lbfgs_pts = _lbfgs_polish(best_X, deadline=start_time + max_time - 5)
        lbfgs_score = _score(lbfgs_pts)
        if lbfgs_score > best_val:
            best_X = lbfgs_pts
            best_val = lbfgs_score
    
    # Ensure validity
    if best_X is None or not np.all(np.isfinite(best_X)) or best_X.shape != (N, 2):
        best_X = _grid_seed(0.06)
    
    return best_X
