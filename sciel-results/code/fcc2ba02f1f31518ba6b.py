"""SA global search + scipy L-BFGS-B penalty polish for 50 equal circles in unit square."""
import numpy as np
import time
import scipy.optimize as opt

N = 50
HALF = 0.5


def _score(c):
    """Largest common radius for centres c (N,2)."""
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    d = c[:, None, :] - c[None, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    min_bound = np.min(HALF - np.abs(c))
    return min(min_pair, min_bound)


def _init_radius(pts):
    """Compute the current radius once for a configuration."""
    d = pts[:, None, :] - pts[None, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    min_bound = np.min(HALF - np.abs(pts))
    return min(min_pair, min_bound)


def _delta_radius(pts, idx, new_pts):
    """
    Incremental radius update: pts differs from new_pts only at indices idx.
    Returns the new radius without recomputing all O(N^2) pairs.
    """
    old = pts[idx]
    new = new_pts[idx]
    k = len(idx)

    # Wall distances for moved points
    wall_new = np.minimum(HALF - np.abs(new[:, 0]), HALF - np.abs(new[:, 1]))
    min_wall_moved = np.min(wall_new) if k else HALF

    # Pair distances involving at least one moved point
    # (i in idx, j in idx): pairs among the moved
    if k > 1:
        d = new[:, None, :] - new[None, :, :]
        dist = np.sqrt(np.sum(d * d, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair_moved = np.min(dist) / 2.0
    else:
        min_pair_moved = np.inf

    # (i in idx, j not in idx): pairs between moved and unmoved
    if k:
        others = np.array([i for i in range(N) if i not in set(idx)])
        if len(others):
            d = new[:, None, :] - pts[others][None, :, :]
            dist = np.sqrt(np.sum(d * d, axis=2))
            min_pair_cross = np.min(dist) / 2.0
        else:
            min_pair_cross = np.inf
    else:
        min_pair_cross = np.inf

    # Unmoved-unmoved pairs are unchanged; we need their min too.
    # To avoid recomputing, cache is complex; instead if k is small, the
    # global min is min(moved-involving, unmoved-unmoved). We track the
    # unmoved-unmoved min via a cheap full recompute only when needed.
    # For safety and simplicity, recompute the unmoved-unmoved min.
    if k:
        others = np.array([i for i in range(N) if i not in set(idx)])
        if len(others) > 1:
            d = pts[others][:, None, :] - pts[others][None, :, :]
            dist = np.sqrt(np.sum(d * d, axis=2))
            np.fill_diagonal(dist, np.inf)
            min_pair_unmoved = np.min(dist) / 2.0
        else:
            min_pair_unmoved = np.inf
    else:
        # nothing moved: full recompute
        d = pts[:, None, :] - pts[None, :]
        dist = np.sqrt(np.sum(d * d, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair_unmoved = np.min(dist) / 2.0

    min_pair = min(min_pair_moved, min_pair_cross, min_pair_unmoved)
    # wall for unmoved unchanged
    min_wall = min(min_wall_moved, np.min(HALF - np.abs(pts[np.array([i for i in range(N) if i not in set(idx)])])) if k else min_wall_moved)
    return min(min_pair, min_wall)


def _sa_search(pts0, deadline, rng):
    """
    Simulated annealing global search. Returns best (pts, radius).
    """
    pts = pts0.copy()
    cur_r = _init_radius(pts)
    best_pts = pts.copy()
    best_r = cur_r

    T_start = 0.04
    T_end = 1e-7
    # We run until deadline; pick a cooling that reaches T_end around deadline.
    # Estimate iterations: we don't know speed, so use adaptive: cool by factor
    # each step based on remaining time.
    # Use a fixed number of steps with geometric cooling, but cap by time.
    n_steps = 200000
    alpha = (T_end / T_start) ** (1.0 / n_steps)

    step_base = 0.06

    for step in range(n_steps):
        if time.time() > deadline:
            break
        T = T_start * (alpha ** step)

        # Move 1-3 random circles
        num_movers = int(rng.integers(1, 4))
        idx = rng.choice(N, size=num_movers, replace=False)

        step_size = step_base * (0.4 + 0.6 * (T / T_start))
        delta = rng.normal(0, step_size, (num_movers, 2))

        new_pts = pts.copy()
        new_pts[idx] += delta
        new_pts = np.clip(new_pts, -HALF, HALF)

        new_r = _delta_radius(pts, idx, new_pts)

        if new_r > cur_r:
            pts = new_pts
            cur_r = new_r
            if cur_r > best_r:
                best_r = cur_r
                best_pts = pts.copy()
        else:
            delta_r = new_r - cur_r
            if delta_r < -12 * T:
                prob = 0.0
            else:
                prob = np.exp(delta_r / T)
            if rng.random() < prob:
                pts = new_pts
                cur_r = new_r

    return best_pts, best_r


def _penalty_obj(x, r_target):
    """Objective for L-BFGS-B: sum of squared constraint violations for radius r_target."""
    centers = x.reshape(-1, 2)
    wall_x = HALF - np.abs(centers[:, 0])
    wall_y = HALF - np.abs(centers[:, 1])
    walls = np.minimum(wall_x, wall_y)
    wall_viol = np.maximum(0.0, r_target - walls)

    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_viol = np.maximum(0.0, 2.0 * r_target - dists)

    return np.sum(wall_viol ** 2) + np.sum(pair_viol ** 2)


def _is_feasible(centers, r_target, tol=1e-9):
    wall_x = HALF - np.abs(centers[:, 0])
    wall_y = HALF - np.abs(centers[:, 1])
    walls = np.minimum(wall_x, wall_y)
    if np.min(walls) < r_target - tol:
        return False
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 2.0 * r_target - tol


def _lbfgs_polish(pts, r_target, deadline, max_iter=400):
    """
    Use L-BFGS-B to try to make configuration feasible for radius r_target.
    Returns (feasible, best_centers, best_radius).
    """
    x0 = pts.reshape(-1).copy()
    bounds = [(-HALF, HALF)] * (2 * N)

    def obj(x):
        return _penalty_obj(x, r_target)

    res = opt.minimize(
        obj, x0, method='L-BFGS-B', bounds=bounds,
        options={'maxiter': max_iter, 'maxfun': max_iter * 3, 'disp': False}
    )
    centers = res.x.reshape(-1, 2)
    centers = np.clip(centers, -HALF, HALF)
    r = _score(centers)
    feasible = _is_feasible(centers, r_target, tol=1e-8)
    return feasible, centers, r


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
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    if len(grid) >= N:
        d2 = np.sum(grid * grid, axis=1)
        idx = np.argsort(d2)[:N]
        grid = grid[idx]
    return np.clip(grid, -HALF + r, HALF - r)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    rng = np.random.default_rng(1234)

    # --- Phase 1: global SA search from a few seeds, keep best ---
    seeds = [_hex_seed(0.071), _grid_seed(0.071)]
    seeds.append(_hex_seed(0.071) + rng.normal(0, 0.01, (N, 2)))
    seeds.append(_grid_seed(0.071) + rng.normal(0, 0.01, (N, 2)))

    best_pts = None
    best_r = 0.0
    # Split SA time among seeds
    n_seeds = len(seeds)
    sa_budget = deadline - 40.0  # reserve 40s for polish
    per_seed = max(1.0, (sa_budget - start_time) / n_seeds)

    for s in seeds:
        s = np.clip(s, -HALF, HALF)
        r0 = _init_radius(s)
        if best_pts is None or r0 > best_r:
            best_pts = s.copy()
            best_r = r0
        if time.time() > deadline - 40.0:
            break
        sa_deadline = min(time.time() + per_seed, deadline - 40.0)
        p, r = _sa_search(s, sa_deadline, rng)
        if r > best_r:
            best_r = r
            best_pts = p

    # --- Phase 2: L-BFGS-B penalty polish to push radius up ---
    if time.time() < deadline - 3.0:
        # Try to improve beyond current best_r by targeting slightly larger radii
        cur = best_pts.copy()
        cur_r = best_r
        for target_mult in [1.0, 1.005, 1.01, 1.02, 1.03, 1.05]:
            r_target = cur_r * target_mult
            if time.time() > deadline - 3.0:
                break
            feasible, centers, r = _lbfgs_polish(cur, r_target, deadline, max_iter=300)
            if r > cur_r:
                cur_r = r
                cur = centers
            if feasible:
                # feasible at r_target; try to go a bit higher
                continue
            else:
                # not feasible; the polish still may have improved radius
                break

        if cur_r > best_r:
            best_r = cur_r
            best_pts = cur

    # --- Phase 3: a final SA micro-search from the best to squeeze more ---
    if time.time() < deadline - 2.0:
        p, r = _sa_search(best_pts, deadline - 1.0, rng)
        if r > best_r:
            best_r = r
            best_pts = p

    return best_pts
