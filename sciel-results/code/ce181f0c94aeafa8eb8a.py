import numpy as np
import time

N = 50
DEADLINE = 140.0  # safety margin below 150s

def _radius_score(centers):
    """Compute the radius score for a given set of centers."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    d_left = centers[:, 0] + 0.5
    d_right = 0.5 - centers[:, 0]
    d_bottom = centers[:, 1] + 0.5
    d_top = 0.5 - centers[:, 1]
    min_boundary = np.min(np.min(np.stack([d_left, d_right, d_bottom, d_top]), axis=0))
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_pairwise = np.min(dist)
    return min(min_boundary, min_pairwise / 2.0)

def _push_apart(centers, r_target, iters=50):
    """Push overlapping circles apart; keep inside square."""
    centers = centers.copy()
    for _ in range(iters):
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                d = centers[i] - centers[j]
                dist = np.linalg.norm(d)
                min_dist = 2.0 * r_target
                if dist < min_dist - 1e-12:
                    if dist < 1e-14:
                        d = np.array([0.001, 0.001])
                        dist = np.linalg.norm(d)
                    push = (min_dist - dist) / 2.0
                    d_hat = d / dist
                    centers[i] += push * d_hat
                    centers[j] -= push * d_hat
                    moved = True
        # Clamp to square
        centers[:, 0] = np.clip(centers[:, 0], -0.5 + r_target, 0.5 - r_target)
        centers[:, 1] = np.clip(centers[:, 1], -0.5 + r_target, 0.5 - r_target)
        if not moved:
            break
    return centers

def _local_polish(centers, r_target, iters=200):
    """Coordinate-descent: nudge each circle to increase the min-separation."""
    centers = centers.copy()
    best_r = r_target
    for _ in range(iters):
        improved = False
        for i in range(N):
            best_pos = centers[i].copy()
            best_local_r = best_r
            # Try small nudges in 4 directions
            for dx, dy in [(0.001, 0), (-0.001, 0), (0, 0.001), (0, -0.001),
                           (0.0005, 0.0005), (-0.0005, 0.0005), (0.0005, -0.0005), (-0.0005, -0.0005)]:
                trial = centers.copy()
                trial[i] = centers[i] + np.array([dx, dy])
                # Check bounds
                if np.any(trial[i] < -0.5) or np.any(trial[i] > 0.5):
                    continue
                # Quick min-separation check
                diffs = trial[i] - trial
                dists = np.linalg.norm(diffs, axis=1)
                dists[i] = np.inf
                min_pair = np.min(dists) / 2.0
                d_left = trial[i, 0] + 0.5
                d_right = 0.5 - trial[i, 0]
                d_bottom = trial[i, 1] + 0.5
                d_top = 0.5 - trial[i, 1]
                min_bound = min(d_left, d_right, d_bottom, d_top)
                local_r = min(min_pair, min_bound)
                if local_r > best_local_r + 1e-14:
                    best_local_r = local_r
                    best_pos = trial[i].copy()
                    improved = True
            centers[i] = best_pos
            best_r = max(best_r, best_local_r)
        if not improved:
            break
    return centers, best_r

def _make_grid_template(rows, cols, jitter=0.0, rng=None):
    """Generate a grid of N points in [-0.5, 0.5]^2."""
    pts = []
    for i in range(rows):
        for j in range(cols):
            if len(pts) >= N:
                break
            x = (j + 0.5) / cols - 0.5
            y = (i + 0.5) / rows - 0.5
            pts.append([x, y])
    pts = np.array(pts[:N])
    if jitter > 0 and rng is not None:
        pts += rng.normal(0, jitter, size=pts.shape)
    return np.clip(pts, -0.49, 0.49)

def _make_hex_template(spacing, rng=None):
    """Hexagonal packing template."""
    pts = []
    row = 0
    y = -0.48
    while len(pts) < N and y < 0.48:
        offset = 0.0 if row % 2 == 0 else spacing * 0.5
        x = -0.48 + offset
        while x < 0.48 and len(pts) < N:
            pts.append([x, y])
            x += spacing
        y += spacing * np.sqrt(3) / 2
        row += 1
    if len(pts) < N:
        # Fill remaining with random
        if rng is None:
            rng = np.random.default_rng(0)
        while len(pts) < N:
            pts.append(rng.uniform(-0.45, 0.45, size=2).tolist())
    return np.clip(np.array(pts[:N]), -0.49, 0.49)

def _solve_from_template(centers, r_target):
    """Given a template and target radius, repair and polish."""
    # First: push apart to achieve target radius
    centers_repaired = _push_apart(centers, r_target, iters=80)
    # Check if we can achieve r_target
    r_actual = _radius_score(centers_repaired)
    if r_actual < r_target - 1e-6:
        # Try to polish from current position
        centers_polished, r_polished = _local_polish(centers_repaired, r_actual, iters=100)
        return centers_polished, r_polished
    else:
        # Polish to squeeze more
        centers_polished, r_polished = _local_polish(centers_repaired, r_actual, iters=150)
        return centers_polished, r_polished

def _binary_search_radius(centers, r_lo, r_hi, max_iters=12):
    """Binary search on radius using repair as feasibility test."""
    best_centers = centers.copy()
    best_r = r_lo
    for _ in range(max_iters):
        r_mid = (r_lo + r_hi) / 2
        c_rep, r_ach = _solve_from_template(centers, r_mid)
        if r_ach >= r_mid - 1e-7:
            best_r = r_ach
            best_centers = c_rep
            r_lo = r_mid
        else:
            r_hi = r_mid
    return best_centers, best_r

def _construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    rng = np.random.default_rng(123)

    best_centers = None
    best_radius = -np.inf

    # Generate diverse templates
    templates = []

    # Square grids
    for rows, cols in [(7, 8), (8, 7), (5, 10), (10, 5), (6, 9), (9, 6)]:
        templates.append(_make_grid_template(rows, cols, rng=rng))

    # Hexagonal with various spacings
    for sp in [0.10, 0.11, 0.12, 0.13, 0.14]:
        templates.append(_make_hex_template(sp, rng=rng))

    # Jittered grids
    for rows, cols in [(7, 8), (8, 7)]:
        for jit in [0.01, 0.02, 0.03]:
            templates.append(_make_grid_template(rows, cols, jitter=jit, rng=rng))

    # Random
    for _ in range(5):
        templates.append(rng.uniform(-0.45, 0.45, size=(N, 2)))

    # Process each template
    for idx, tmpl in enumerate(templates):
        if time.time() > deadline - 15:
            break

        # Estimate initial radius from template
        diff = tmpl[:, np.newaxis, :] - tmpl[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_dist = np.min(dist)
        d_left = tmpl[:, 0] + 0.5
        d_right = 0.5 - tmpl[:, 0]
        d_bottom = tmpl[:, 1] + 0.5
        d_top = 0.5 - tmpl[:, 1]
        min_bound = np.min(np.min(np.stack([d_left, d_right, d_bottom, d_top]), axis=0))
        r_est = min(min_dist / 2.0, min_bound)

        # Try to improve: binary search around estimated radius
        r_lo = max(0.0, r_est - 0.005)
        r_hi = r_est + 0.01

        c_sol, r_sol = _binary_search_radius(tmpl, r_lo, r_hi, max_iters=10)

        if r_sol > best_radius:
            best_radius = r_sol
            best_centers = c_sol

    # Final polish on best
    if best_centers is not None and time.time() < deadline - 5:
        c_pol, r_pol = _local_polish(best_centers, best_radius, iters=200)
        if r_pol > best_radius:
            best_radius = r_pol
            best_centers = c_pol

    # Fallback
    if best_centers is None:
        best_centers = templates[0]

    return best_centers

def construct():
    return _construct()
