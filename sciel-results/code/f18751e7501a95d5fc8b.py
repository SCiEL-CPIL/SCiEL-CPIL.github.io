import numpy as np
import time
from scipy.spatial.distance import cdist

N = 200
HALF = 0.5
_TIME_LIMIT = 135  # Leave 15s margin for 150s kill


def _score(c):
    """Return the common radius for a configuration (higher is better)."""
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # inside square
    if np.any(np.abs(c) > HALF + 1e-12):
        return 0.0
    # pairwise min distance
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    # wall distance
    wall = np.min([
        np.min(c[:, 0]) + HALF,
        HALF - np.max(c[:, 0]),
        np.min(c[:, 1]) + HALF,
        HALF - np.max(c[:, 1]),
    ])
    r = min(wall, min_pair / 2.0)
    return r if r > 0 else 0.0


def _hex_grid_points(offset_x=0.0, offset_y=0.0, scale=1.0):
    """
    Generate a hexagonal lattice of points covering the unit square plus margin.
    Returns array of shape (M, 2) for M points.
    """
    spacing = 1.0 / 16.0 * scale
    dx = spacing
    dy = spacing * np.sqrt(3) / 2.0
    
    margin = 0.1
    x_min, x_max = -HALF - margin, HALF + margin
    y_min, y_max = -HALF - margin, HALF + margin
    
    nx = int(np.ceil((x_max - x_min) / dx)) + 1
    ny = int(np.ceil((y_max - y_min) / dy)) + 1
    
    points = []
    for j in range(ny):
        y = y_min + j * dy + offset_y
        x_offset = 0.0 if j % 2 == 0 else dx / 2.0
        for i in range(nx):
            x = x_min + i * dx + x_offset + offset_x
            points.append([x, y])
    
    return np.array(points)


def _greedy_select_fast(candidate_points, n_select=N):
    """
    Faster greedy: use a priority-queue-like approach with vectorized updates.
    At each step, pick the candidate with the largest min(dist_to_selected/2, wall_dist).
    """
    mask = np.all(np.abs(candidate_points) <= HALF, axis=1)
    candidates = candidate_points[mask]
    
    if len(candidates) < n_select:
        extra = np.random.uniform(-HALF, HALF, (n_select - len(candidates), 2))
        candidates = np.vstack([candidates, extra])
    
    n_cand = len(candidates)
    
    wall_dists = np.min([
        candidates[:, 0] + HALF,
        HALF - candidates[:, 0],
        candidates[:, 1] + HALF,
        HALF - candidates[:, 1],
    ], axis=0)
    
    current_min_dist = np.full(n_cand, np.inf)
    selected = np.zeros(n_select, dtype=int)
    used = np.zeros(n_cand, dtype=bool)
    
    for step in range(n_select):
        score = np.minimum(current_min_dist / 2.0, wall_dists)
        score[used] = -1.0
        
        best_idx = np.argmax(score)
        selected[step] = best_idx
        used[best_idx] = True
        
        if step < n_select - 1:
            d_new = np.linalg.norm(candidates - candidates[best_idx], axis=1)
            current_min_dist = np.minimum(current_min_dist, d_new)
            current_min_dist[used] = np.inf
    
    return candidates[selected]


def _soft_min_value_and_grad(pts, beta):
    """
    Compute the soft-min value and its gradient for the circle packing problem.
    Returns (value, grad) where grad has shape (N, 2).
    """
    n = len(pts)
    d_mat = cdist(pts, pts)
    np.fill_diagonal(d_mat, np.inf)
    
    # Wall distances
    w_left = pts[:, 0] + HALF
    w_right = HALF - pts[:, 0]
    w_bottom = pts[:, 1] + HALF
    w_top = HALF - pts[:, 1]
    
    # Pairwise constraints: d_ij / 2
    v_pairs = d_mat / 2.0
    # Wall constraints
    v_walls = np.stack([w_left, w_right, w_bottom, w_top], axis=1)  # (N, 4)
    
    # Compute log-sum-exp for stability
    # For pairs: log_sum_exp(-beta * v_pairs)
    # For walls: log_sum_exp(-beta * v_walls)
    
    # Flatten for computation
    vp_flat = v_pairs[~np.eye(n, dtype=bool)]
    vw_flat = v_walls.ravel()
    
    all_vals = np.concatenate([vp_flat, vw_flat])
    
    # Log-sum-exp
    max_val = np.max(all_vals)
    if max_val < -1e10:
        return 0.0, np.zeros((n, 2))
    
    log_sum = max_val + np.log(np.sum(np.exp(all_vals - max_val)))
    value = -log_sum / beta
    
    # Compute weights
    weights_pairs = np.exp(-beta * v_pairs)
    np.fill_diagonal(weights_pairs, 0.0)
    weights_walls = np.exp(-beta * v_walls)
    
    Z = np.sum(weights_pairs) + np.sum(weights_walls)
    if Z < 1e-10:
        return value, np.zeros((n, 2))
    
    # Gradients
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        grad_pair = diff / (2.0 * d_mat[:, :, np.newaxis])
    grad_pair[np.eye(n, dtype=bool), :] = 0.0
    
    grad_walls = np.zeros((n, 4, 2))
    grad_walls[:, 0, 0] = 1.0
    grad_walls[:, 1, 0] = -1.0
    grad_walls[:, 2, 1] = 1.0
    grad_walls[:, 3, 1] = -1.0
    
    grad_total = (np.sum(weights_pairs[:, :, np.newaxis] * grad_pair, axis=1) +
                  np.sum(weights_walls[:, :, np.newaxis] * grad_walls, axis=1)) / Z
    
    return value, grad_total


def _gradient_polish(pts, max_iters=200, beta=50.0, lr_init=0.01):
    """
    Gradient ascent on the soft-min of constraints.
    """
    pts = pts.copy()
    n = len(pts)
    
    for it in range(max_iters):
        value, grad = _soft_min_value_and_grad(pts, beta)
        
        lr = lr_init / (1.0 + it / 20.0)
        pts += lr * grad
        pts = np.clip(pts, -HALF, HALF)
        
        if time.time() > _TIME_LIMIT - 5:
            break
    
    return pts


def _local_search(pts, max_iters=500, step_size=0.01):
    """
    Simple local search: try small random perturbations and keep improvements.
    """
    pts = pts.copy()
    current_r = _score(pts)
    n = len(pts)
    
    for it in range(max_iters):
        if time.time() > _TIME_LIMIT - 5:
            break
        
        # Pick a random point and perturb it
        idx = np.random.randint(n)
        delta = np.random.uniform(-step_size, step_size, 2)
        new_pt = pts[idx].copy()
        new_pt += delta
        new_pt = np.clip(new_pt, -HALF, HALF)
        
        pts[idx] = new_pt
        new_r = _score(pts)
        
        if new_r <= current_r:
            pts[idx] -= delta  # revert
        else:
            current_r = new_r
    
    return pts


def _overlap_energy(pts, k=5):
    """
    Compute a smooth overlap energy for global exploration.
    This is a surrogate that is easier to optimize than the hard min constraint.
    """
    n = len(pts)
    d_mat = cdist(pts, pts)
    np.fill_diagonal(d_mat, np.inf)
    
    # For each pair, if distance < 2r (where r is current min), add penalty
    # Use a smooth approximation
    # Energy = sum_{i<j} max(0, 2r - d_ij)^2
    # But we don't know r yet. Use a fixed target or iterate.
    
    # Simpler: use the k-nearest neighbors
    # For each point, find its k nearest neighbors
    # Energy = sum over points of sum over k-nearest of (d_ij - target)^2
    # where target is some desired spacing
    
    # Actually, let's use a different approach:
    # Compute the minimum distance for each point to its nearest neighbor
    # MinDist_i = min_{j != i} d_ij
    # Energy = sum_i (MinDist_i - target)^2 + wall penalties
    
    # But this is still complex. Let's use a simpler energy:
    # E = sum_{i<j} exp(-d_ij / sigma) + wall penalties
    # This encourages points to be far apart.
    
    sigma = 0.05
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = d_mat[i, j]
            energy += np.exp(-d / sigma)
    
    # Wall penalties
    for i in range(n):
        energy += np.exp(-(pts[i, 0] + HALF) / sigma)
        energy += np.exp(-(HALF - pts[i, 0]) / sigma)
        energy += np.exp(-(pts[i, 1] + HALF) / sigma)
        energy += np.exp(-(HALF - pts[i, 1]) / sigma)
    
    return energy


def _gradient_descent_overlap(pts, max_iters=100, lr=0.005, k=5):
    """
    Gradient descent on the overlap energy for global exploration.
    """
    pts = pts.copy()
    n = len(pts)
    sigma = 0.05
    
    for it in range(max_iters):
        if time.time() > _TIME_LIMIT - 10:
            break
        
        d_mat = cdist(pts, pts)
        np.fill_diagonal(d_mat, np.inf)
        
        # Gradient of exp(-d/sigma) w.r.t. x_i
        # d/dx_i exp(-d_ij/sigma) = -exp(-d_ij/sigma) / sigma * (x_i - x_j) / d_ij
        
        grad = np.zeros((n, 2))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = d_mat[i, j]
                if d < 1e-10:
                    continue
                factor = -np.exp(-d / sigma) / sigma / d
                grad[i] += factor * (pts[i] - pts[j])
            
            # Wall gradients
            grad[i, 0] += -np.exp(-(pts[i, 0] + HALF) / sigma) / sigma
            grad[i, 0] += -np.exp(-(HALF - pts[i, 0]) / sigma) / sigma
            grad[i, 1] += -np.exp(-(pts[i, 1] + HALF) / sigma) / sigma
            grad[i, 1] += -np.exp(-(HALF - pts[i, 1]) / sigma) / sigma
        
        pts -= lr * grad
        pts = np.clip(pts, -HALF, HALF)
    
    return pts


def construct():
    t0 = time.time()
    best_r = 0.0
    best_pts = None
    
    np.random.seed(42)
    
    # Phase 1: Generate initial configurations from hex grids with different offsets
    offsets_x = [0.0, 0.02, 0.04, 0.01, 0.03]
    offsets_y = [0.0, 0.02, 0.04, 0.01, 0.03]
    scales = [1.0, 0.95, 1.05]
    
    candidates = []
    attempts = 0
    max_attempts = 20
    
    for scale in scales:
        for ox in offsets_x:
            for oy in offsets_y:
                if time.time() - t0 > _TIME_LIMIT - 30:
                    break
                if attempts >= max_attempts:
                    break
                
                grid = _hex_grid_points(offset_x=ox, offset_y=oy, scale=scale)
                pts = _greedy_select_fast(grid)
                attempts += 1
                
                # Quick polish
                if time.time() - t0 < _TIME_LIMIT - 20:
                    pts = _gradient_polish(pts, max_iters=50, beta=50.0, lr_init=0.01)
                
                r = _score(pts)
                if r > best_r:
                    best_r = r
                    best_pts = pts.copy()
                
                candidates.append((r, pts.copy()))
        
        if time.time() - t0 > _TIME_LIMIT - 30:
            break
    
    # Phase 2: For the top few candidates, do deeper optimization
    candidates.sort(key=lambda x: -x[0])
    top_k = min(3, len(candidates))
    
    for i in range(top_k):
        if time.time() - t0 > _TIME_LIMIT - 10:
            break
        
        pts = candidates[i][1].copy()
        
        # Try overlap energy descent for global exploration
        if time.time() - t0 < _TIME_LIMIT - 15:
            pts_explored = _gradient_descent_overlap(pts, max_iters=50, lr=0.003)
            r_explored = _score(pts_explored)
            if r_explored > best_r:
                best_r = r_explored
                best_pts = pts_explored.copy()
        
        # Gradient polish
        if time.time() - t0 < _TIME_LIMIT - 10:
            pts_polished = _gradient_polish(pts, max_iters=100, beta=100.0, lr_init=0.005)
            r_polished = _score(pts_polished)
            if r_polished > best_r:
                best_r = r_polished
                best_pts = pts_polished.copy()
        
        # Local search
        if time.time() - t0 < _TIME_LIMIT - 5:
            pts_local = _local_search(best_pts, max_iters=200, step_size=0.005)
            r_local = _score(pts_local)
            if r_local > best_r:
                best_r = r_local
                best_pts = pts_local.copy()
    
    # Final polish
    if best_pts is not None and time.time() - t0 < _TIME_LIMIT - 2:
        final = _gradient_polish(best_pts, max_iters=100, beta=200.0, lr_init=0.002)
        r_final = _score(final)
        if r_final > best_r:
            best_r = r_final
            best_pts = final.copy()
    
    if best_pts is None:
        best_pts = np.random.uniform(-HALF, HALF, (N, 2))
    
    return best_pts
