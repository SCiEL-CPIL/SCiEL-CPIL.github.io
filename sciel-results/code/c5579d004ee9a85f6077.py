import numpy as np
from scipy.spatial.distance import cdist
import time

N = 200
_TIME_LIMIT = 145  # Leave 5s margin for 150s kill


def _score(pts):
    """Compute the maximum radius for a set of N points in [-0.5, 0.5]^2."""
    if pts.shape[0] != N or not np.all(np.isfinite(pts)):
        return 0.0
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D)
    wall = np.min([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1]),
    ])
    r = min(min_pair / 2.0, wall)
    return r if r > 0 else 0.0


def _hex_grid_points(nrows, ncols, offset_x=0.0, offset_y=0.0):
    """
    Generate a hexagonal lattice of points.
    Returns all points (may be more than N).
    """
    pts = []
    # Spacing: for a hex grid with spacing s, rows are offset by s/2
    # We want to cover [-0.5, 0.5]^2
    # Let's use spacing based on ncols
    s_x = 1.0 / ncols  # horizontal spacing
    s_y = s_x * np.sqrt(3) / 2  # vertical spacing for hex
    # But we need nrows * ncols ≈ 200, so let's just generate a grid
    # and pick the best N
    
    # Actually, let's generate a dense hex grid and then select
    # Number of rows and cols to get roughly N points
    # For hex: each row has ncols or ncols-1 points
    # Let's just generate a rectangular grid with hex offset
    
    for i in range(nrows):
        y = -0.49 + i * (0.98 / max(nrows - 1, 1)) + offset_y
        shift = (s_x / 2) if (i % 2 == 1) else 0.0
        for j in range(ncols):
            x = -0.49 + j * (0.98 / max(ncols - 1, 1)) + shift + offset_x
            pts.append([x, y])
    
    pts = np.array(pts)
    return pts


def _greedy_select(pts, N_target):
    """
    Greedy selection: start with the point closest to center,
    then repeatedly add the point that maximizes the minimum distance
    to already-selected points.
    """
    if len(pts) <= N_target:
        return pts.copy()
    
    selected = []
    remaining = list(range(len(pts)))
    
    # Start with point closest to origin
    d2_origin = np.sum(pts ** 2, axis=1)
    first = np.argmin(d2_origin)
    selected.append(first)
    remaining.remove(first)
    
    # Precompute distances from first point
    min_dists = np.full(len(pts), np.inf)
    min_dists[first] = 0
    
    for _ in range(N_target - 1):
        # Find point in remaining with max min_dist
        # min_dists for remaining points
        best_idx = -1
        best_val = -1
        for idx in remaining:
            if min_dists[idx] > best_val:
                best_val = min_dists[idx]
                best_idx = idx
        
        if best_idx == -1:
            break
        
        selected.append(best_idx)
        remaining.remove(best_idx)
        
        # Update min_dists for remaining points
        d = np.linalg.norm(pts[best_idx] - pts, axis=1)
        for idx in remaining:
            if d[idx] < min_dists[idx]:
                min_dists[idx] = d[idx]
    
    return pts[selected]


def _greedy_select_fast(pts, N_target):
    """
    Faster greedy selection using vectorized operations.
    """
    if len(pts) <= N_target:
        return pts.copy()
    
    n = len(pts)
    selected = []
    remaining_mask = np.ones(n, dtype=bool)
    
    # Start with point closest to origin
    d2_origin = np.sum(pts ** 2, axis=1)
    first = np.argmin(d2_origin)
    selected.append(first)
    remaining_mask[first] = False
    
    # min_dist for each point to the set of selected points
    min_dists = np.linalg.norm(pts - pts[first], axis=1)
    
    for _ in range(N_target - 1):
        # Among remaining, find max min_dist
        # Set selected points' min_dist to -inf
        min_dists_sel = min_dists.copy()
        min_dists_sel[~remaining_mask] = -np.inf
        best_idx = np.argmax(min_dists_sel)
        
        if not remaining_mask[best_idx]:
            break
        
        selected.append(best_idx)
        remaining_mask[best_idx] = False
        
        # Update min_dists
        d = np.linalg.norm(pts - pts[best_idx], axis=1)
        min_dists = np.minimum(min_dists, d)
    
    return pts[selected]


def _local_refine(pts, n_iters=2000, lr_schedule='decay'):
    """
    Local refinement: iteratively find the bottleneck (closest pair or wall)
    and adjust points to increase the minimum radius.
    """
    pts = pts.copy()
    
    for it in range(n_iters):
        if time.time() > _TIME_LIMIT - 5:
            break
        
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)
        
        # Find min pair
        min_pair_idx = np.unravel_index(np.argmin(D), D.shape)
        i, j = min_pair_idx
        d_pair = D[i, j]
        
        # Find min wall distance
        wall_dists = np.array([
            np.min(pts[:, 0]) + 0.5,
            0.5 - np.max(pts[:, 0]),
            np.min(pts[:, 1]) + 0.5,
            0.5 - np.max(pts[:, 1]),
        ])
        min_wall_val = np.min(wall_dists)
        min_wall_idx = np.argmin(wall_dists)
        
        # Learning rate
        if lr_schedule == 'decay':
            lr = 0.01 / (1 + it * 0.01)
        else:
            lr = 0.005
        
        # Determine which constraint is limiting
        if d_pair / 2.0 <= min_wall_val:
            # Push pair apart
            diff = pts[j] - pts[i]
            norm = np.linalg.norm(diff)
            if norm < 1e-10:
                pts[i, 0] += 0.001
                pts[j, 0] -= 0.001
            else:
                unit = diff / norm
                step = lr * min(1.0, 1.0 / norm)
                pts[i] -= step * unit
                pts[j] += step * unit
        else:
            # Push wall particle inward
            if min_wall_idx == 0:  # left wall
                idx = np.argmin(pts[:, 0])
                pts[idx, 0] += lr
            elif min_wall_idx == 1:  # right wall
                idx = np.argmax(pts[:, 0])
                pts[idx, 0] -= lr
            elif min_wall_idx == 2:  # bottom wall
                idx = np.argmin(pts[:, 1])
                pts[idx, 1] += lr
            else:  # top wall
                idx = np.argmax(pts[:, 1])
                pts[idx, 1] -= lr
        
        # Clamp to bounds
        pts[:, 0] = np.clip(pts[:, 0], -0.4999, 0.4999)
        pts[:, 1] = np.clip(pts[:, 1], -0.4999, 0.4999)
    
    return pts


def _hill_climb(pts, n_restarts=1000):
    """
    Hill climbing: try small random perturbations and keep if they improve score.
    """
    pts = pts.copy()
    best_r = _score(pts)
    
    for _ in range(n_restarts):
        if time.time() > _TIME_LIMIT - 5:
            break
        
        # Random perturbation
        noise = np.random.normal(0, 0.005, size=pts.shape)
        new_pts = pts + noise
        new_pts[:, 0] = np.clip(new_pts[:, 0], -0.4999, 0.4999)
        new_pts[:, 1] = np.clip(new_pts[:, 1], -0.4999, 0.4999)
        
        new_r = _score(new_pts)
        if new_r > best_r:
            pts = new_pts
            best_r = new_r
    
    return pts, best_r


def construct():
    t0 = time.time()
    best_r = 0.0
    best_pts = None
    
    # Try multiple hex grid configurations with different offsets
    # Each configuration: (nrows, ncols, offset_x, offset_y)
    configs = []
    
    # Base configurations
    base_configs = [
        (14, 15),
        (15, 14),
        (13, 16),
        (16, 13),
        (12, 17),
        (17, 12),
        (11, 18),
        (18, 11),
    ]
    
    # Add offsets
    offsets = [
        (0.0, 0.0),
        (0.01, 0.0),
        (-0.01, 0.0),
        (0.0, 0.01),
        (0.0, -0.01),
        (0.01, 0.01),
        (-0.01, -0.01),
        (0.02, 0.0),
        (0.0, 0.02),
    ]
    
    for nrows, ncols in base_configs:
        for ox, oy in offsets:
            configs.append((nrows, ncols, ox, oy))
    
    # Shuffle configs for diversity
    np.random.seed(42)
    np.random.shuffle(configs)
    
    for nrows, ncols, ox, oy in configs:
        if time.time() - t0 > _TIME_LIMIT - 30:
            break
        
        # Generate hex grid points
        all_pts = _hex_grid_points(nrows, ncols, ox, oy)
        
        # Greedy select N points
        selected = _greedy_select_fast(all_pts, N)
        
        # Local refine
        refined = _local_refine(selected, n_iters=500)
        r = _score(refined)
        
        if r > best_r:
            best_r = r
            best_pts = refined.copy()
        
        # Hill climb for a bit
        if time.time() - t0 < _TIME_LIMIT - 15:
            hc_pts, hc_r = _hill_climb(refined, n_restarts=200)
            if hc_r > best_r:
                best_r = hc_r
                best_pts = hc_pts.copy()
    
    # If we have time, do more hill climbing on the best solution
    if best_pts is not None and time.time() - t0 < _TIME_LIMIT - 10:
        hc_pts, hc_r = _hill_climb(best_pts, n_restarts=500)
        if hc_r > best_r:
            best_r = hc_r
            best_pts = hc_pts.copy()
    
    # Final local refinement
    if best_pts is not None and time.time() - t0 < _TIME_LIMIT - 5:
        final_pts = _local_refine(best_pts, n_iters=300)
        r = _score(final_pts)
        if r > best_r:
            best_r = r
            best_pts = final_pts
    
    if best_pts is None:
        # Fallback: simple grid
        rows, cols = 14, 15
        xs = np.linspace(-0.45, 0.45, cols)
        ys = np.linspace(-0.45, 0.45, rows)
        grid = np.array([(x, y) for y in ys for x in xs])
        best_pts = grid[:N]
    
    return best_pts
