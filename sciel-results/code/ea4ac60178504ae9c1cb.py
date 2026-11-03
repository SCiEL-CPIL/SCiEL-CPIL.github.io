"""Pack 200 equal circles in a unit square to maximize common radius."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

N = 200
_START_TIME = time.time()
_TIME_LIMIT = 135  # seconds, leave margin before 150s kill


def _compute_radius(pts):
    """Compute the achievable radius for a set of points."""
    if len(pts) < N:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)
    if np.any(~np.isfinite(pts)):
        return 0.0
    # Check inside square
    if np.any(pts[:, 0] < -0.5) or np.any(pts[:, 0] > 0.5) or \
       np.any(pts[:, 1] < -0.5) or np.any(pts[:, 1] > 0.5):
        return 0.0
    # Min distance to wall
    wall = np.min([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1]),
    ])
    # Min pairwise distance
    from scipy.spatial.distance import cdist
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D)
    r = min(wall, min_pair / 2.0)
    return r if r > 0 else 0.0


def _soft_min_radius(pts, k=50.0):
    """
    Smooth surrogate for the radius using log-sum-exp soft-min.
    r_soft = -1/k * log( sum_i exp(-k * d_i) )  where d_i are the constraints.
    This is differentiable and approximates min(d_i) as k -> inf.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if np.any(~np.isfinite(pts)):
        return -1e10
    if np.any(pts[:, 0] < -0.5) or np.any(pts[:, 0] > 0.5) or \
       np.any(pts[:, 1] < -0.5) or np.any(pts[:, 1] > 0.5):
        return -1e10
    # Wall distances
    wall_dists = np.array([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1]),
    ])
    # Pairwise distances / 2
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    pair_dists = D / 2.0
    # Combine all constraints
    all_dists = np.concatenate([wall_dists, pair_dists.ravel()])
    # Clip to avoid overflow in exp
    all_dists = np.clip(all_dists, -1.0, 10.0)
    # Soft min via log-sum-exp with better numerical stability
    neg_k_d = -k * all_dists
    # Shift for numerical stability
    m = np.max(neg_k_d)
    if m > 500:
        m = 500
    exp_terms = np.exp(neg_k_d - m)
    log_sum = m + np.log(np.sum(exp_terms))
    r_soft = -log_sum / k
    return r_soft


def _epigraph_objective(params, k=50.0):
    """
    Epigraph formulation: maximize t subject to all constraints >= t.
    params = [x1, y1, x2, y2, ..., xN, yN, t]
    We minimize -t + penalty for constraint violations.
    """
    n = len(params)
    t = params[-1]
    pts = params[:-1].reshape(-1, 2)
    
    # Penalty for points outside square
    outside_pen = 0.0
    if np.any(pts[:, 0] < -0.5) or np.any(pts[:, 0] > 0.5) or \
       np.any(pts[:, 1] < -0.5) or np.any(pts[:, 1] > 0.5):
        outside_pen = 10.0 * (np.sum(np.maximum(0, pts[:, 0] - 0.5)**2) +
                              np.sum(np.maximum(0, -0.5 - pts[:, 0])**2) +
                              np.sum(np.maximum(0, pts[:, 1] - 0.5)**2) +
                              np.sum(np.maximum(0, -0.5 - pts[:, 1])**2))
    
    # Wall constraint violations: t > wall_dist
    wall_dists = np.array([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1]),
    ])
    wall_viol = np.maximum(0, t - wall_dists)
    
    # Pairwise constraint violations: t > dist/2
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    pair_dists = D / 2.0
    pair_viol = np.maximum(0, t - pair_dists)
    
    # Soft penalty using log-sum-exp
    all_viol = np.concatenate([wall_viol, pair_viol.ravel()])
    if np.max(all_viol) > 0.1:
        # Hard penalty for large violations
        penalty = 100.0 * np.sum(all_viol**2)
    else:
        # Smooth penalty
        k_eff = k
        neg_k_v = -k_eff * all_viol
        m = np.max(neg_k_v)
        if m > 500:
            m = 500
        log_sum = m + np.log(np.sum(np.exp(neg_k_v - m)))
        penalty = -log_sum / k_eff
    
    return -t + penalty + outside_pen


def _push_apart(pts, n_iters=100, strength=0.005):
    """
    Iteratively push apart the closest pairs of points.
    This is a simple local search that directly addresses the bottleneck
    of the minimum pairwise distance.
    """
    pts = pts.copy().astype(np.float64)
    for it in range(n_iters):
        # Find the closest pair
        from scipy.spatial.distance import cdist
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)
        min_idx = np.unravel_index(np.argmin(D), D.shape)
        i, j = min_idx
        min_dist = D[i, j]
        if min_dist < 1e-12:
            # Coincident points; separate them randomly
            pts[j] += np.random.randn(2) * 0.01
            continue
        # Push apart
        direction = pts[j] - pts[i]
        direction /= np.linalg.norm(direction)
        # Check if pushing would violate boundaries
        move = strength * direction
        new_i = pts[i] - move
        new_j = pts[j] + move
        # Check boundaries
        if -0.5 <= new_i[0] <= 0.5 and -0.5 <= new_i[1] <= 0.5 and \
           -0.5 <= new_j[0] <= 0.5 and -0.5 <= new_j[1] <= 0.5:
            pts[i] = new_i
            pts[j] = new_j
        else:
            # Try smaller move
            for frac in [0.5, 0.25, 0.1]:
                move = strength * frac * direction
                new_i = pts[i] - move
                new_j = pts[j] + move
                if -0.5 <= new_i[0] <= 0.5 and -0.5 <= new_i[1] <= 0.5 and \
                   -0.5 <= new_j[0] <= 0.5 and -0.5 <= new_j[1] <= 0.5:
                    pts[i] = new_i
                    pts[j] = new_j
                    break
    return pts


def _local_refine_soft(pts, k=50.0, maxiter=300):
    """
    Use scipy.optimize.minimize with a smooth soft-min objective.
    This is more efficient than Nelder-Mead on the non-smooth min.
    """
    pts = pts.copy().astype(np.float64)
    
    def _neg_soft(params):
        p = params.reshape(-1, 2)
        return -_soft_min_radius(p, k=k)
    
    x0 = pts.flatten()
    try:
        res = minimize(_neg_soft, x0, method='L-BFGS-B',
                      options={'maxiter': maxiter, 'gtol': 1e-8, 'ftol': 1e-12})
        new_pts = res.x.reshape(-1, 2)
        # Verify validity
        if np.all(new_pts[:, 0] >= -0.5) and np.all(new_pts[:, 0] <= 0.5) and \
           np.all(new_pts[:, 1] >= -0.5) and np.all(new_pts[:, 1] <= 0.5):
            r = _compute_radius(new_pts)
            if r > 0:
                return new_pts, r
    except Exception:
        pass
    return pts, _compute_radius(pts)


def _force_repulsion(pts, n_iters=500, strength=0.002, cooling=0.995):
    """
    Force-directed repulsion: each pair of circles exerts a repulsive force
    when closer than 2*r. Also apply attraction to walls when too close.
    This is a physics-based local search.
    """
    pts = pts.copy().astype(np.float64)
    r_current = _compute_radius(pts)
    if r_current <= 0:
        return pts, 0.0
    
    for it in range(n_iters):
        # Compute forces
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)
        
        # Repulsive forces from pairs closer than 2*r
        force = np.zeros_like(pts)
        mask = D < 2.0 * r_current
        for i in range(N):
            for j in range(i + 1, N):
                if mask[i, j]:
                    d = D[i, j]
                    if d < 1e-10:
                        # Random push for coincident points
                        force[i] -= np.random.randn(2) * 0.01
                        force[j] += np.random.randn(2) * 0.01
                    else:
                        direction = pts[j] - pts[i]
                        direction /= d
                        # Force magnitude: stronger when closer
                        f_mag = strength * (2.0 * r_current - d) / (2.0 * r_current)
                        force[i] -= f_mag * direction
                        force[j] += f_mag * direction
        
        # Wall forces: push away from walls if too close
        for axis in range(2):
            # Left wall
            too_close_left = pts[:, axis] < -0.5 + r_current
            if np.any(too_close_left):
                force[too_close_left, axis] += strength * (r_current - (pts[too_close_left, axis] + 0.5))
            # Right wall
            too_close_right = pts[:, axis] > 0.5 - r_current
            if np.any(too_close_right):
                force[too_close_right, axis] -= strength * (r_current - (0.5 - pts[too_close_right, axis]))
        
        # Apply forces with cooling
        step_size = strength * (cooling ** it)
        pts_new = pts + step_size * force
        
        # Clamp to square
        pts_new[:, 0] = np.clip(pts_new[:, 0], -0.5, 0.5)
        pts_new[:, 1] = np.clip(pts_new[:, 1], -0.5, 0.5)
        
        # Check if this improves the radius
        r_new = _compute_radius(pts_new)
        if r_new > r_current:
            pts = pts_new
            r_current = r_new
        else:
            # Try smaller step
            for frac in [0.5, 0.25, 0.1]:
                pts_try = pts + step_size * frac * force
                pts_try[:, 0] = np.clip(pts_try[:, 0], -0.5, 0.5)
                pts_try[:, 1] = np.clip(pts_try[:, 1], -0.5, 0.5)
                r_try = _compute_radius(pts_try)
                if r_try > r_current:
                    pts = pts_try
                    r_current = r_try
                    break
    
    return pts, r_current


def _best_result():
    """Fallback: 10x20 rectangular grid."""
    rows, cols = 20, 10
    dx = 1.0 / cols
    dy = 1.0 / rows
    xs = -0.5 + (np.arange(cols) + 0.5) * dx
    ys = -0.5 + (np.arange(rows) + 0.5) * dy
    grid = np.array([(x, y) for y in ys for x in xs])
    return grid[:N]


def _greedy_lattice_select(grid_size, stagger=True, time_limit=None):
    """
    Greedy best-fit selection from a fine lattice of candidate sites.
    Select N points maximizing the minimum distance to already-selected neighbors
    and the square boundaries.
    
    Parameters:
    - grid_size: number of divisions per side for the candidate lattice
    - stagger: if True, use a hexagonal/staggered lattice; if False, square lattice
    - time_limit: max seconds for this greedy pass
    
    Returns: (points_array, radius)
    """
    if time_limit is None:
        time_limit = 20.0
    t0 = time.time()
    
    # Generate candidate lattice sites
    if stagger:
        # Hexagonal lattice
        dx = 1.0 / grid_size
        dy = dx * np.sqrt(3) / 2.0
        # Adjust to fit in square
        n_rows = int(1.0 / dy) + 2
        n_cols = grid_size + 2
        candidates = []
        for i in range(n_rows):
            y = -0.5 + i * dy
            if y < -0.5 or y > 0.5:
                continue
            shift = dx / 2.0 if i % 2 == 1 else 0.0
            for j in range(n_cols):
                x = -0.5 + shift + j * dx
                if x >= -0.5 and x <= 0.5:
                    candidates.append((x, y))
        candidates = np.array(candidates)
    else:
        # Square lattice
        dx = 1.0 / grid_size
        dy = 1.0 / grid_size
        xs = np.linspace(-0.5, 0.5, grid_size + 1)
        ys = np.linspace(-0.5, 0.5, grid_size + 1)
        candidates = np.array([(x, y) for y in ys for x in xs])
    
    n_cand = len(candidates)
    if n_cand < N:
        return None, 0.0
    
    # Greedy selection: at each step, pick the candidate that maximizes
    # the minimum distance to already-selected points AND to the walls.
    selected = []
    # Start with corners or center for better initialization
    # Use a simple heuristic: start with the point farthest from all walls
    # Actually, let's just do pure greedy from empty set
    
    # Precompute distance to walls for each candidate
    wall_dist = np.min([
        candidates[:, 0] + 0.5,
        0.5 - candidates[:, 0],
        candidates[:, 1] + 0.5,
        0.5 - candidates[:, 1],
    ], axis=0)
    
    # Track minimum distance to selected points for each candidate
    min_dist_to_selected = np.full(n_cand, np.inf)
    
    for step in range(N):
        if time.time() - t0 > time_limit:
            break
        # Score for each candidate: min(wall_dist, min_dist_to_selected)
        # But we want to maximize the final radius, which is the min over all selected
        # Greedy: pick the candidate with the largest current "bottleneck"
        score = np.minimum(wall_dist, min_dist_to_selected)
        # Pick the best candidate
        idx = np.argmax(score)
        selected.append(idx)
        
        # Update min_dist_to_selected for remaining candidates
        if step < N - 1:
            new_pt = candidates[idx]
            # Compute distances from new_pt to all other candidates
            d = np.sqrt(np.sum((candidates - new_pt) ** 2, axis=1))
            d[idx] = np.inf
            min_dist_to_selected = np.minimum(min_dist_to_selected, d)
    
    if len(selected) < N:
        return None, 0.0
    
    pts = candidates[selected[:N]]
    r = _compute_radius(pts)
    return pts, r


def _rect_grid_score(params, rows, cols):
    """Score (radius) for a rectangular grid. params = [dx, dy, x0, y0].
    We constrain circles to be inside, so radius = min(dx/2, dy/2, dist_to_wall).
    Actually, for a grid, the radius is limited by half the minimum spacing and
    the minimum distance to a wall. Let's compute it directly.
    """
    dx, dy, x0, y0 = params
    if dx <= 0 or dy <= 0:
        return -1e10
    # Generate points
    xs = x0 + np.arange(cols) * dx
    ys = y0 + np.arange(rows) * dy
    pts = np.array([(x, y) for y in ys for x in xs])[:N]
    # Check all inside [-0.5, 0.5]^2
    if np.any(pts[:, 0] < -0.5) or np.any(pts[:, 0] > 0.5) or \
       np.any(pts[:, 1] < -0.5) or np.any(pts[:, 1] > 0.5):
        return -1e10
    # Min distance to wall
    wall = np.min([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1]),
    ])
    # Min pairwise distance
    if N > 1:
        # Use cdist for speed
        from scipy.spatial.distance import cdist
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)
        min_pair = np.min(D)
    else:
        min_pair = np.inf
    r = min(wall, min_pair / 2)
    if r <= 0:
        return -1e10
    return r


def _hex_grid_score(params, rows, cols):
    """Score for a hexagonal (staggered) grid.
    params = [dx, dy, x0, y0, offset]
    offset: shift of odd rows (0 = no stagger, 0.5*dx = full stagger)
    """
    dx, dy, x0, y0, off = params
    if dx <= 0 or dy <= 0:
        return -1e10
    pts = []
    for i in range(rows):
        y = y0 + i * dy
        shift = off * dx if i % 2 == 1 else 0.0
        xs = x0 + shift + np.arange(cols) * dx
        for x in xs:
            pts.append((x, y))
    pts = np.array(pts[:N])
    if len(pts) < N:
        return -1e10
    if np.any(pts[:, 0] < -0.5) or np.any(pts[:, 0] > 0.5) or \
       np.any(pts[:, 1] < -0.5) or np.any(pts[:, 1] > 0.5):
        return -1e10
    wall = np.min([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1]),
    ])
    from scipy.spatial.distance import cdist
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D)
    r = min(wall, min_pair / 2)
    if r <= 0:
        return -1e10
    return r


def _neg_rect_score(params, rows, cols):
    return -_rect_grid_score(params, rows, cols)


def _neg_hex_score(params, rows, cols):
    return -_hex_grid_score(params, rows, cols)


def construct():
    t0 = time.time()
    best_r = 0.0
    best_pts = _best_result()
    
    # First, try greedy lattice selection with various grid sizes
    # Use more time for greedy since it's a good baseline
    greedy_time_budget = 40.0
    for grid_size in [15, 20, 25, 30, 40]:
        if time.time() - t0 > _TIME_LIMIT - greedy_time_budget - 20:
            break
        for stagger in [True, False]:
            if time.time() - t0 > _TIME_LIMIT - greedy_time_budget - 20:
                break
            try:
                pts, r = _greedy_lattice_select(grid_size, stagger, time_limit=8.0)
                if pts is not None and r > best_r:
                    best_r = r
                    best_pts = pts
            except Exception:
                pass

    # Try rectangular grids with various row/col splits
    # 200 = rows * cols, rows >= cols (tall grids) or we can also try wide
    factor_pairs = []
    for c in range(1, int(np.sqrt(N)) + 1):
        if N % c == 0:
            r = N // c
            factor_pairs.append((r, c))
            factor_pairs.append((c, r))
    # Also add near-factor pairs for flexibility (not required to be exact factors,
    # but we'll only use exact for simplicity)
    factor_pairs = list(set(factor_pairs))

    # Also try non-factor splits by truncating/padding? No, keep exact.
    # But we can also try rows*cols > N and just use N points from the grid.
    # Let's add some additional splits.
    extra = [(14, 15), (15, 14), (20, 10), (10, 20), (25, 8), (8, 25),
             (16, 13), (13, 16), (12, 17), (17, 12), (20, 10), (10, 20)]
    for pair in extra:
        if pair not in factor_pairs:
            factor_pairs.append(pair)
    
    # Sort by expected quality (more square-like is better)
    factor_pairs.sort(key=lambda p: abs(p[0] - p[1]))

    # Rectangular grids
    for (rows, cols) in factor_pairs:
        if time.time() - t0 > _TIME_LIMIT:
            break
        if rows * cols < N:
            continue
        # Initial guess: uniform grid
        dx0 = 1.0 / cols
        dy0 = 1.0 / rows
        x00 = -0.5 + dx0 / 2
        y00 = -0.5 + dy0 / 2
        x0 = np.array([dx0, dy0, x00, y00])
        try:
            res = minimize(_neg_rect_score, x0, args=(rows, cols), method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            r = -res.fun
            if r > best_r:
                best_r = r
                dx, dy, x0v, y0v = res.x
                xs = x0v + np.arange(cols) * dx
                ys = y0v + np.arange(rows) * dy
                pts = np.array([(x, y) for y in ys for x in xs])[:N]
                if np.all(pts[:, 0] >= -0.5) and np.all(pts[:, 0] <= 0.5) and \
                   np.all(pts[:, 1] >= -0.5) and np.all(pts[:, 1] <= 0.5):
                    best_pts = pts
        except Exception:
            pass

    # Hexagonal grids
    for (rows, cols) in factor_pairs:
        if time.time() - t0 > _TIME_LIMIT:
            break
        if rows * cols < N:
            continue
        dx0 = 1.0 / cols
        dy0 = dx0 * np.sqrt(3) / 2
        # Adjust so it fits: total height = (rows-1)*dy0, need <= 1
        if (rows - 1) * dy0 > 1.0:
            dy0 = 1.0 / (rows - 1) if rows > 1 else 1.0
        x00 = -0.5 + dx0 / 2
        y00 = -0.5 + dy0 / 2
        off0 = 0.5  # full stagger
        x0 = np.array([dx0, dy0, x00, y00, off0])
        try:
            res = minimize(_neg_hex_score, x0, args=(rows, cols), method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            r = -res.fun
            if r > best_r:
                best_r = r
                dx, dy, x0v, y0v, offv = res.x
                pts = []
                for i in range(rows):
                    y = y0v + i * dy
                    shift = offv * dx if i % 2 == 1 else 0.0
                    xs = x0v + shift + np.arange(cols) * dx
                    for x in xs:
                        pts.append((x, y))
                pts = np.array(pts[:N])
                if len(pts) == N and \
                   np.all(pts[:, 0] >= -0.5) and np.all(pts[:, 0] <= 0.5) and \
                   np.all(pts[:, 1] >= -0.5) and np.all(pts[:, 1] <= 0.5):
                    best_pts = pts
        except Exception:
            pass

    # Final polish: multiple strategies
    if best_r > 0:
        # Strategy 1: Force-directed repulsion (physics-based)
        if time.time() - t0 < _TIME_LIMIT - 30:
            try:
                pts_force, r_force = _force_repulsion(best_pts, n_iters=300, strength=0.003, cooling=0.997)
                if r_force > best_r:
                    best_r = r_force
                    best_pts = pts_force
            except Exception:
                pass
        
        # Strategy 2: Push-apart local search
        if time.time() - t0 < _TIME_LIMIT - 25:
            try:
                pts_pushed = _push_apart(best_pts, n_iters=200, strength=0.003)
                r_pushed = _compute_radius(pts_pushed)
                if r_pushed > best_r:
                    best_r = r_pushed
                    best_pts = pts_pushed
            except Exception:
                pass
        
        # Strategy 3: Soft-min optimization with L-BFGS-B
        if time.time() - t0 < _TIME_LIMIT - 15:
            try:
                pts_soft, r_soft = _local_refine_soft(best_pts, k=50.0, maxiter=300)
                if r_soft > best_r:
                    best_r = r_soft
                    best_pts = pts_soft
            except Exception:
                pass
        
        # Strategy 4: Another force repulsion pass
        if time.time() - t0 < _TIME_LIMIT - 10:
            try:
                pts_force2, r_force2 = _force_repulsion(best_pts, n_iters=200, strength=0.002, cooling=0.998)
                if r_force2 > best_r:
                    best_r = r_force2
                    best_pts = pts_force2
            except Exception:
                pass
        
        # Strategy 5: Final soft-min with higher k for sharper approximation
        if time.time() - t0 < _TIME_LIMIT - 5:
            try:
                pts_soft2, r_soft2 = _local_refine_soft(best_pts, k=100.0, maxiter=200)
                if r_soft2 > best_r:
                    best_r = r_soft2
                    best_pts = pts_soft2
            except Exception:
                pass
        
        # Strategy 6: Very fine-tuned final push-apart
        if time.time() - t0 < _TIME_LIMIT - 2:
            try:
                pts_pushed3 = _push_apart(best_pts, n_iters=50, strength=0.001)
                r_pushed3 = _compute_radius(pts_pushed3)
                if r_pushed3 > best_r:
                    best_r = r_pushed3
                    best_pts = pts_pushed3
            except Exception:
                pass
    
    if best_r <= 0:
        return _best_result()
    return best_pts
