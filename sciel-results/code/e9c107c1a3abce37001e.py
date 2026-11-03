import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE_MARGIN = 10.0  # seconds to spare before 120s limit


def _compute_violations_and_score(x, y, r, w):
    """Compute constraint violations and score for a packing.
    Returns (total_violation, sum_r).
    """
    h = 2.0 - w
    if w <= 0.05 or h <= 0.05:
        return 1e10, 0.0

    # Boundary violations
    bx = np.maximum(r - x, 0) + np.maximum(x - (w - r), 0)
    by = np.maximum(r - y, 0) + np.maximum(y - (h - r), 0)
    boundary_viol = np.sum(bx) + np.sum(by)

    # Overlap violations
    diff_x = x[:, None] - x[None, :]
    diff_y = y[:, None] - y[None, :]
    dist = np.sqrt(diff_x**2 + diff_y**2)
    np.fill_diagonal(dist, np.inf)
    sum_r = r[:, None] + r[None, :]
    overlap = np.maximum(sum_r - dist, 0)
    overlap_viol = np.sum(overlap)

    total_viol = boundary_viol + overlap_viol
    return total_viol, np.sum(r)


def _objective(vars, w_fixed=None):
    """Objective function for optimizer. Minimizes violations - sum_r."""
    if w_fixed is not None:
        w = w_fixed
        pts_r = vars[:3 * N].reshape(N, 3)
        x, y, r = pts_r[:, 0], pts_r[:, 1], np.maximum(pts_r[:, 2], 1e-8)
    else:
        w = vars[3 * N]
        pts_r = vars[:3 * N].reshape(N, 3)
        x, y, r = pts_r[:, 0], pts_r[:, 1], np.maximum(pts_r[:, 2], 1e-8)

    h = 2.0 - w
    if w <= 0.05 or h <= 0.05:
        return 1e10

    # Boundary violations (squared for smoothness)
    bx = np.maximum(r - x, 0)**2 + np.maximum(x - (w - r), 0)**2
    by = np.maximum(r - y, 0)**2 + np.maximum(y - (h - r), 0)**2
    boundary_viol = np.sum(bx) + np.sum(by)

    # Overlap violations (squared)
    diff_x = x[:, None] - x[None, :]
    diff_y = y[:, None] - y[None, :]
    dist_sq = diff_x**2 + diff_y**2
    dist = np.sqrt(dist_sq)
    np.fill_diagonal(dist, np.inf)
    sum_r_ij = r[:, None] + r[None, :]
    overlap = np.maximum(sum_r_ij - dist, 0)**2
    overlap_viol = np.sum(overlap)

    penalty = 100.0 * (boundary_viol + overlap_viol)
    return penalty - np.sum(r)


def _greedy_place(w, h, n_circles):
    """Greedy placement: place circles one by one at the position with most room."""
    placed = []
    # Try to place circles of decreasing radius
    # Start with a reasonable radius estimate
    base_r = 0.5 / max(3, int(np.sqrt(n_circles))) * 0.9

    for i in range(n_circles):
        r = base_r * (1.0 - 0.02 * i)  # slightly decreasing
        best_pos = None
        best_score = -np.inf

        # Grid search for best position
        n_grid = 15
        for gx in range(n_grid):
            for gy in range(n_grid):
                cx = (gx + 0.5) / n_grid * w
                cy = (gy + 0.5) / n_grid * h

                # Check if circle fits in bounds
                if cx < r or cx > w - r or cy < r or cy > h - r:
                    continue

                # Check distance to all placed circles
                min_dist = np.inf
                for px, py, pr in placed:
                    d = np.sqrt((cx - px)**2 + (cy - py)**2)
                    if d < r + pr:
                        min_dist = d
                        break

                if min_dist < r:
                    # Overlapping, compute how bad
                    score = min_dist - r
                else:
                    # Non-overlapping, score by clearance
                    clearance = min_dist - r
                    # Also consider distance to walls
                    wall_clear = min(cx - r, w - r - cx, cy - r, h - r - cy)
                    score = clearance + 0.5 * wall_clear

                if score > best_score:
                    best_score = score
                    best_pos = (cx, cy, r)

        if best_pos is not None:
            placed.append(best_pos)
        else:
            # Fallback: place at center with small radius
            placed.append((w / 2, h / 2, r * 0.5))

    return np.array(placed)


def _random_start(w, h, seed=None):
    """Generate a random initial placement."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    # Estimate max radius
    max_r = min(w, h) / 3.0

    x = rng.uniform(0, w, N)
    y = rng.uniform(0, h, N)
    r = rng.uniform(0.05, max_r * 0.7, N)

    return x, y, r


def _pack_for_aspect(w, max_iters=200, seed=0):
    """Try to pack circles for a given aspect ratio. Returns best (score, circles) or None."""
    h = 2.0 - w
    if w <= 0.1 or h <= 0.1:
        return None

    # Try multiple initializations
    best_score = -np.inf
    best_circles = None

    inits = []

    # Greedy initialization
    greedy = _greedy_place(w, h, N)
    inits.append(greedy)

    # Random initializations
    for s in range(3):
        x, y, r = _random_start(w, h, seed=seed + s * 100)
        inits.append(np.column_stack((x, y, r)))

    # Grid initialization
    cols = max(2, int(np.round(np.sqrt(N * w / h))))
    rows = max(2, int(np.ceil(N / cols)))
    grid = []
    for i in range(cols):
        for j in range(rows):
            if len(grid) < N:
                gx = (i + 0.5) / cols * w
                gy = (j + 0.5) / rows * h
                gr = 0.45 / max(cols, rows) * min(w / cols, h / rows)
                grid.append([gx, gy, gr])
    inits.append(np.array(grid))

    for init in inits:
        x0, y0, r0 = init[:, 0], init[:, 1], init[:, 2]

        # Phase 1: Optimize positions and radii with fixed w
        def obj_fixed(vars):
            pts = vars.reshape(N, 3)
            x, y, r = pts[:, 0], pts[:, 1], np.maximum(pts[:, 2], 1e-8)
            return _objective(np.concatenate([vars, [w]]), w_fixed=w)

        try:
            res = minimize(
                obj_fixed,
                init.flatten(),
                method='Nelder-Mead',
                options={'maxiter': max_iters, 'xatol': 1e-5, 'fatol': 1e-5}
            )

            # Check validity
            pts = res.x.reshape(N, 3)
            x, y, r = pts[:, 0], pts[:, 1], np.maximum(pts[:, 2], 0)
            viol, score = _compute_violations_and_score(x, y, r, w)

            if viol < 1e-4 and score > best_score:
                best_score = score
                best_circles = np.column_stack((x, y, r))

        except Exception:
            pass

    return best_score, best_circles if best_circles is not None else None


def construct():
    start_time = time.time()
    deadline = start_time + 120.0 - DEADLINE_MARGIN

    best_score = -np.inf
    best_circles = None

    # Search over aspect ratios
    # Perimeter = 2(w+h) = 4, so h = 2-w
    # w ranges from ~0.2 to ~1.8 (symmetric, so just go up to 1.0 and mirror)
    aspect_ratios = np.linspace(0.25, 1.0, 15)

    for w in aspect_ratios:
        if time.time() > deadline:
            break

        result = _pack_for_aspect(w, max_iters=150, seed=42)
        if result is not None:
            score, circles = result
            if circles is not None and score > best_score:
                best_score = score
                best_circles = circles

    # Also try the symmetric counterparts
    for w in aspect_ratios:
        if time.time() > deadline:
            break
        w_sym = 2.0 - w
        if w_sym <= 0.25 or w_sym >= 1.0:
            continue
        result = _pack_for_aspect(w_sym, max_iters=150, seed=43)
        if result is not None:
            score, circles = result
            if circles is not None and score > best_score:
                best_score = score
                best_circles = circles

    # Refine the best solution with more iterations
    if best_circles is not None and time.time() < deadline - 5:
        # Find the w that was used
        # We need to re-derive w from the best circles
        # Actually, let's just do a final refinement on the best aspect
        # Try a few more iterations on the best aspect ratio
        for w_try in np.linspace(0.3, 0.9, 8):
            if time.time() > deadline:
                break
            h = 2.0 - w_try
            if h <= 0.1:
                continue

            # Use best_circles as starting point, scaled to this aspect
            x0 = best_circles[:, 0] * w_try / 1.0  # rough scaling
            y0 = best_circles[:, 1] * h / 1.0
            r0 = best_circles[:, 2] * 0.9

            # Clip to bounds
            x0 = np.clip(x0, r0, w_try - r0)
            y0 = np.clip(y0, r0, h - r0)

            init = np.column_stack((x0, y0, r0))

            def obj_fixed(vars):
                pts = vars.reshape(N, 3)
                x, y, r = pts[:, 0], pts[:, 1], np.maximum(pts[:, 2], 1e-8)
                return _objective(np.concatenate([vars, [w_try]]), w_fixed=w_try)

            try:
                res = minimize(
                    obj_fixed,
                    init.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                pts = res.x.reshape(N, 3)
                x, y, r = pts[:, 0], pts[:, 1], np.maximum(pts[:, 2], 0)
                viol, score = _compute_violations_and_score(x, y, r, w_try)
                if viol < 1e-4 and score > best_score:
                    best_score = score
                    best_circles = np.column_stack((x, y, r))
            except Exception:
                pass

    # Fallback: simple grid
    if best_circles is None:
        w, h = 1.0, 1.0
        cols, rows = 5, 5
        cells = []
        for i in range(cols):
            for j in range(rows):
                if len(cells) < N:
                    cx = (i + 0.5) / cols * w
                    cy = (j + 0.5) / rows * h
                    cr = 0.45 / max(cols, rows) * min(w / cols, h / rows)
                    cells.append([cx, cy, cr])
        best_circles = np.array(cells)

    return best_circles
