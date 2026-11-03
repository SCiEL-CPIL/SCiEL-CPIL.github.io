"""Pack 200 equal circles in a unit square to maximize common radius."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree
import time

N = 200
_TIME_LIMIT = 142  # seconds, leave safety margin before 150s kill


def _score(points):
    """Compute the radius for a set of points in [-0.5, 0.5]^2."""
    if not np.all(np.isfinite(points)):
        return 0.0
    if not np.all((points >= -0.5) & (points <= 0.5)):
        return 0.0
    if len(points) < 2:
        min_pair = float('inf')
    else:
        d = pdist(points)
        min_pair = np.min(d)
    dist_boundary = 0.5 - np.max(np.abs(points), axis=1)
    min_boundary = np.min(dist_boundary)
    return min(min_pair / 2.0, min_boundary)


def _score_fast(points, k=20):
    """Fast approximate score using k-nearest neighbors."""
    if not np.all(np.isfinite(points)):
        return 0.0
    if not np.all((points >= -0.5) & (points <= 0.5)):
        return 0.0
    n = len(points)
    if n < 2:
        min_pair = float('inf')
    else:
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=min(k, n))
        min_pair = np.min(dists[:, 1:min(k, n)])
    dist_boundary = 0.5 - np.max(np.abs(points), axis=1)
    min_boundary = np.min(dist_boundary)
    return min(min_pair / 2.0, min_boundary)


def _soft_min_score(points, beta=100.0):
    """Soft-min approximation of the radius for gradient-based optimization.
    
    Uses log-sum-exp to approximate min over all pair distances and wall distances.
    """
    if not np.all(np.isfinite(points)):
        return 0.0
    if not np.all((points >= -0.5) & (points <= 0.5)):
        return 0.0
    n = len(points)
    if n < 2:
        min_pair = float('inf')
    else:
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=min(30, n))
        pair_dists = dists[:, 1:min(30, n)]
        # Soft min over pair distances
        # r_pair = (1/beta) * log(sum(exp(-beta * d))) for each point, then min over points
        # Actually, we want the global min. Use a two-level soft min.
        # For efficiency, use the k-NN distances as proxies.
        # Soft min of all values: -1/beta * log(sum(exp(-beta*x)))
        # But we have a matrix. Let's flatten and use a subset.
        # For speed, use the 30 nearest neighbors per point.
        flat_dists = pair_dists.flatten()
        # Clip to avoid overflow
        flat_dists = np.clip(flat_dists, 1e-10, 1.0)
        # Soft min
        max_d = np.max(flat_dists)
        soft_min_pair = max_d - np.log(np.sum(np.exp(-beta * (flat_dists - max_d)))) / beta
    dist_boundary = 0.5 - np.max(np.abs(points), axis=1)
    dist_boundary = np.clip(dist_boundary, 1e-10, 1.0)
    max_b = np.max(dist_boundary)
    soft_min_wall = max_b - np.log(np.sum(np.exp(-beta * (dist_boundary - max_b)))) / beta
    
    return min(soft_min_pair / 2.0, soft_min_wall)


def _best_result():
    """Fallback: 10x20 rectangular grid."""
    rows, cols = 20, 10
    dx = 1.0 / cols
    dy = 1.0 / rows
    xs = -0.5 + (np.arange(cols) + 0.5) * dx
    ys = -0.5 + (np.arange(rows) + 0.5) * dy
    grid = np.array([(x, y) for y in ys for x in xs])
    return grid[:N]


def _generate_hex_grid():
    """Generate a hexagonal lattice packing as initial configuration."""
    best_centers = None
    best_r = 0.0
    
    for n_rows in range(12, 22):
        n_cols = int(np.ceil(N / n_rows))
        
        r1 = 1.0 / (2.0 * n_cols)
        r2 = 1.0 / ((n_rows - 1) * np.sqrt(3) + 2)
        r = min(r1, r2)
        s = 2 * r
        
        centers = []
        for i in range(n_rows):
            for j in range(n_cols):
                if len(centers) >= N:
                    break
                x = -0.5 + r + j * s + (i % 2) * s / 2
                y = -0.5 + r + i * s * np.sqrt(3) / 2
                centers.append([x, y])
            if len(centers) >= N:
                break
        
        centers = np.array(centers[:N])
        
        if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
            xs = centers[:, 0]
            ys = centers[:, 1]
            cx = (xs.max() + xs.min()) / 2
            cy = (ys.max() + ys.min()) / 2
            centers[:, 0] -= cx
            centers[:, 1] -= cy
            
            if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
                continue
        
        r_actual = _score(centers)
        if r_actual > best_r:
            best_r = r_actual
            best_centers = centers.copy()
    
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centers


def _local_refine(points, time_budget):
    """Local refinement using gradient-based optimization with soft-min objective.
    
    Phase 1: L-BFGS-B with soft-min (smooth) objective, increasing beta.
    Phase 2: L-BFGS-B with exact k-NN score for final polish.
    """
    start = time.time()
    current = points.copy()
    current_score = _score(current)
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Phase 1: Soft-min with increasing beta
    betas = [50.0, 100.0, 200.0, 400.0]
    for beta in betas:
        remaining = time_budget - (time.time() - start)
        if remaining < 5:
            break
        
        def _neg_soft(pts, beta=beta):
            pts = pts.reshape(-1, 2)
            if not np.all((pts >= -0.5) & (pts <= 0.5)):
                return 1e10
            if not np.all(np.isfinite(pts)):
                return 1e10
            r = _soft_min_score(pts, beta)
            return -r
        
        try:
            res = minimize(_neg_soft, current.flatten(), method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12,
                                    'maxls': 50})
            new_pts = res.x.reshape(-1, 2)
            new_score = _score(new_pts)
            if new_score > current_score:
                current = new_pts
                current_score = new_score
        except Exception:
            pass
    
    # Phase 2: Exact k-NN score for final polish
    remaining = time_budget - (time.time() - start)
    if remaining > 5:
        def _neg_exact(pts):
            pts = pts.reshape(-1, 2)
            if not np.all((pts >= -0.5) & (pts <= 0.5)):
                return 1e10
            if not np.all(np.isfinite(pts)):
                return 1e10
            r = _score_fast(pts, k=20)
            return -r
        
        try:
            res2 = minimize(_neg_exact, current.flatten(), method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13,
                                    'maxls': 30})
            new_pts = res2.x.reshape(-1, 2)
            new_score = _score(new_pts)
            if new_score > current_score:
                current = new_pts
                current_score = new_score
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead if time remains
    remaining = time_budget - (time.time() - start)
    if remaining > 10:
        def _neg_exact2(pts):
            pts = pts.reshape(-1, 2)
            if not np.all((pts >= -0.5) & (pts <= 0.5)):
                return 1e10
            if not np.all(np.isfinite(pts)):
                return 1e10
            r = _score(pts)
            return -r
        
        try:
            res3 = minimize(_neg_exact2, current.flatten(), method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14,
                                    'adaptive': True})
            new_pts = res3.x.reshape(-1, 2)
            new_score = _score(new_pts)
            if new_score > current_score:
                current = new_pts
                current_score = new_score
        except Exception:
            pass
    
    return current, current_score


def _local_search(centers, time_budget):
    """Force-displacement local search: push apart the most constrained pairs."""
    centers = centers.copy()
    best_r = _score(centers)
    best_centers = centers.copy()
    start = time.time()
    
    step_scale = 0.005
    
    while time.time() - start < time_budget:
        if time.time() - start > time_budget - 2:
            break
        
        tree = cKDTree(centers)
        dists, idx = tree.query(centers, k=3)
        nn_dists = dists[:, 1]
        nn_idxs = idx[:, 1]
        
        wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
        
        # Find most constrained points
        min_pair_idx = np.argmin(nn_dists)
        min_wall_idx = np.argmin(wall_dists)
        
        # Select points to move: those with smallest nn_dist and smallest wall_dist
        candidates = np.unique(np.concatenate([
            np.argsort(nn_dists)[:5],
            np.argsort(wall_dists)[:5]
        ]))
        
        improved = False
        for _ in range(15):
            trial = centers.copy()
            for idx in candidates:
                # Determine push direction
                # If wall-constrained, push away from wall
                # If pair-constrained, push away from nearest neighbor
                if wall_dists[idx] < nn_dists[idx] / 2:
                    # Push away from nearest wall
                    x, y = trial[idx]
                    if x < -0.45:
                        dx, dy = 1, 0
                    elif x > 0.45:
                        dx, dy = -1, 0
                    elif y < -0.45:
                        dx, dy = 0, 1
                    elif y > 0.45:
                        dx, dy = 0, -1
                    else:
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx, dy = np.cos(angle), np.sin(angle)
                else:
                    # Push away from nearest neighbor
                    nb = nn_idxs[idx]
                    dx, dy = trial[idx, 0] - trial[nb, 0], trial[idx, 1] - trial[nb, 1]
                    norm = np.sqrt(dx**2 + dy**2)
                    if norm > 1e-10:
                        dx, dy = dx / norm, dy / norm
                    else:
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx, dy = np.cos(angle), np.sin(angle)
                
                step = step_scale * (0.5 + np.random.rand())
                trial[idx, 0] += step * dx
                trial[idx, 1] += step * dy
            
            trial[:, 0] = np.clip(trial[:, 0], -0.5, 0.5)
            trial[:, 1] = np.clip(trial[:, 1], -0.5, 0.5)
            
            r_trial = _score(trial)
            if r_trial > best_r + 1e-12:
                best_r = r_trial
                best_centers = trial.copy()
                centers = trial.copy()
                improved = True
        
        if not improved:
            step_scale *= 0.95
            if step_scale < 1e-7:
                break
    
    return best_centers, best_r


def construct():
    t0 = time.time()
    
    # Start with hexagonal grid
    best_pts = _generate_hex_grid()
    best_r = _score(best_pts)
    
    # Also try rectangular grid
    rect_pts = _best_result()
    rect_r = _score(rect_pts)
    if rect_r > best_r:
        best_pts = rect_pts
        best_r = rect_r
    
    # Local refinement with soft-min gradient method
    remaining = _TIME_LIMIT - (time.time() - t0)
    if remaining > 15:
        best_pts, best_r = _local_refine(best_pts, remaining - 10)
    
    # Force-displacement local search
    remaining = _TIME_LIMIT - (time.time() - t0)
    if remaining > 10:
        best_pts, best_r = _local_search(best_pts, remaining - 5)
    
    # Final verification and clamping
    best_pts = np.clip(best_pts, -0.5, 0.5)
    final_r = _score(best_pts)
    if final_r > best_r:
        best_r = final_r
    
    if best_r <= 0:
        return _best_result()
    return best_pts
