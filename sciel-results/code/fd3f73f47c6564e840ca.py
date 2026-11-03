"""Rewritten program for circle packing in perimeter-4 rectangle."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 21

# EVOLVE-BLOCK-START

def _compute_value(circles):
    """Compute sum of radii and check validity. Returns (value, valid)."""
    if circles is None or len(circles) != N:
        return 0.0, False
    if not np.all(np.isfinite(circles)):
        return 0.0, False
    radii = circles[:, 2]
    if np.any(radii <= 0):
        return 0.0, False
    if np.any(radii > 1.0):
        return 0.0, False

    min_x = np.min(circles[:, 0] - radii)
    max_x = np.max(circles[:, 0] + radii)
    min_y = np.min(circles[:, 1] - radii)
    max_y = np.max(circles[:, 1] + radii)

    W = max_x - min_x
    H = max_y - min_y

    if W <= 0 or H <= 0:
        return 0.0, False
    if 2 * (W + H) > 4.0 + 1e-9:
        return 0.0, False

    cx = circles[:, 0]
    cy = circles[:, 1]
    r = radii
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((cx[i] - cx[j])**2 + (cy[i] - cy[j])**2)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False

    return np.sum(radii), True


def _penalty_objective(params, W, H, penalty_weight=500.0):
    """
    Objective: negative sum of radii + penalty for overlaps and boundary violations.
    params: [x_0, y_0, r_0, ..., x_{N-1}, y_{N-1}, r_{N-1}]
    """
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]

    # Boundary violations (smooth)
    bx_pen = 0.0
    for i in range(N):
        bx_pen += max(0.0, -(xs[i] - rs[i]))**2
        bx_pen += max(0.0, (xs[i] + rs[i]) - W)**2
        bx_pen += max(0.0, -(ys[i] - rs[i]))**2
        bx_pen += max(0.0, (ys[i] + rs[i]) - H)**2

    # Overlap violations (smooth)
    ov_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx*dx + dy*dy
            min_dist = rs[i] + rs[j]
            if dist_sq < min_dist*min_dist:
                dist = np.sqrt(dist_sq)
                overlap = min_dist - dist
                ov_pen += overlap*overlap

    total_penalty = penalty_weight * (bx_pen + ov_pen)
    return -np.sum(rs) + total_penalty


def _grid_init(W, H, n_cols, n_rows):
    """Create initial grid placement for N circles."""
    circles = []
    for i in range(n_cols):
        for j in range(n_rows):
            if len(circles) >= N:
                break
            cx = (i + 0.5) * W / n_cols
            cy = (j + 0.5) * H / n_rows
            cell_w = W / n_cols
            cell_h = H / n_rows
            r = min(cell_w, cell_h) * 0.42
            circles.append([cx, cy, r])
    while len(circles) < N:
        circles.append([W/2, H/2, min(W,H)*0.04])
    return np.array(circles[:N])


def _random_init(W, H, rng):
    """Random initial placement."""
    circles = []
    for _ in range(N):
        cx = rng.uniform(0.1, W-0.1)
        cy = rng.uniform(0.1, H-0.1)
        r = rng.uniform(0.02, min(W,H)*0.15)
        circles.append([cx, cy, r])
    return np.array(circles)


def _polish(params, W, H, time_budget, rng):
    """Polish a candidate with L-BFGS-B, then Nelder-Mead for local refinement."""
    start = time.time()
    best_params = params.copy()
    best_obj = _penalty_objective(params, W, H)

    # L-BFGS-B pass
    bounds = []
    for i in range(N):
        r_max = min(W, H) / 2
        bounds.append((1e-6, W))
        bounds.append((1e-6, H))
        bounds.append((1e-6, r_max))

    try:
        res = minimize(_penalty_objective, best_params, args=(W, H),
                       method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-9})
        if res.fun < best_obj:
            best_obj = res.fun
            best_params = res.x.copy()
    except Exception:
        pass

    # Nelder-Mead pass
    if time.time() - start < time_budget * 0.6:
        try:
            res2 = minimize(_penalty_objective, best_params, args=(W, H),
                            method='Nelder-Mead',
                            options={'maxiter': 800, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res2.fun < best_obj:
                best_obj = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass

    # Second L-BFGS-B polish
    if time.time() - start < time_budget:
        try:
            res3 = minimize(_penalty_objective, best_params, args=(W, H),
                            method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
            if res3.fun < best_obj:
                best_obj = res3.fun
                best_params = res3.x.copy()
        except Exception:
            pass

    return best_params


def _validate_and_extract(params, W, H):
    """Extract circles from params, clamp, and validate."""
    circles = params.reshape(N, 3).copy()
    circles[:, 0] = np.clip(circles[:, 0], circles[:, 2], W - circles[:, 2])
    circles[:, 1] = np.clip(circles[:, 1], circles[:, 2], H - circles[:, 2])
    circles[:, 2] = np.clip(circles[:, 2], 1e-6, min(W, H)/2)

    val, valid = _compute_value(circles)
    if valid:
        return val, circles
    return 0.0, None


def _pack_for_aspect(W, H, time_budget, rng):
    """Run multiple strategies for a given aspect ratio."""
    start = time.time()
    best_val = 0.0
    best_circles = None

    # Strategy 1: Differential evolution (global search)
    if time.time() - start < time_budget * 0.4:
        try:
            bounds = []
            for i in range(N):
                r_max = min(W, H) / 2
                bounds.append((1e-4, W))
                bounds.append((1e-4, H))
                bounds.append((1e-4, r_max))

            # Use grid init as seed
            grid_circles = _grid_init(W, H, 5, 5)
            seed = grid_circles.flatten()

            de_result = differential_evolution(
                _penalty_objective, bounds, args=(W, H),
                seed=seed,
                maxiter=80, popsize=20,
                tol=1e-8, mutation=(0.3, 1.0), recombination=0.8,
                polish=False,
                init='latinhypercube'
            )
            if de_result.fun < 0:  # negative means sum of radii > penalty
                val, circles = _validate_and_extract(de_result.x, W, H)
                if val > best_val:
                    best_val = val
                    best_circles = circles
        except Exception:
            pass

    # Strategy 2: Grid-based local optimization with multiple restarts
    remaining_time = time_budget - (time.time() - start)
    if remaining_time > 2:
        grid_configs = [(5, 5), (3, 7), (7, 3), (4, 6), (6, 4)]
        for nc, nr in grid_configs:
            if time.time() - start > time_budget * 0.8:
                break
            circles_init = _grid_init(W, H, nc, nr)
            params = circles_init.flatten()
            t_left = time_budget - (time.time() - start)
            if t_left < 1:
                break
            polished = _polish(params, W, H, min(t_left, 5.0), rng)
            val, circles = _validate_and_extract(polished, W, H)
            if val > best_val:
                best_val = val
                best_circles = circles

    # Strategy 3: Random restarts with local optimization
    remaining_time = time_budget - (time.time() - start)
    n_random = max(1, int(remaining_time / 3.0))
    for k in range(n_random):
        if time.time() - start > time_budget * 0.95:
            break
        circles_init = _random_init(W, H, rng)
        params = circles_init.flatten()
        t_left = time_budget - (time.time() - start)
        if t_left < 1:
            break
        polished = _polish(params, W, H, min(t_left, 4.0), rng)
        val, circles = _validate_and_extract(polished, W, H)
        if val > best_val:
            best_val = val
            best_circles = circles

    return best_val, best_circles


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    rng = np.random.default_rng(12345)

    best_val = 0.0
    best_circles = None

    # Try a range of aspect ratios (W+H = 2)
    aspect_ratios = [
        (1.0, 1.0),
        (1.2, 0.8),
        (0.8, 1.2),
        (1.4, 0.6),
        (0.6, 1.4),
        (1.3, 0.7),
        (0.7, 1.3),
        (1.1, 0.9),
        (0.9, 1.1),
        (1.5, 0.5),
        (0.5, 1.5),
    ]

    # Sort by closeness to square (square tends to pack better)
    aspect_ratios.sort(key=lambda wh: abs(wh[0]/wh[1] - 1.0))

    n_aspects = len(aspect_ratios)
    for idx, (W, H) in enumerate(aspect_ratios):
        if time.time() > deadline:
            break

        # Allocate time: more to earlier (more promising) aspects
        remaining = deadline - time.time()
        remaining_aspects = n_aspects - idx
        time_per_aspect = max(3.0, remaining / max(1, remaining_aspects))
        time_per_aspect = min(time_per_aspect, 20.0)  # Cap per aspect

        val, circles = _pack_for_aspect(W, H, time_per_aspect, rng)
        if val > best_val:
            best_val = val
            best_circles = circles.copy()

    # Fallback if nothing valid
    if best_circles is None:
        W, H = 1.0, 1.0
        best_circles = _grid_init(W, H, 5, 5)
        val, valid = _compute_value(best_circles)
        if not valid:
            side = 5
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.35 / side)
                     for i in range(side) for j in range(side)]
            best_circles = np.array(cells[:N])

    # Final validation
    val, valid = _compute_value(best_circles)
    if not valid:
        W, H = 1.0, 1.0
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.33 / side)
                 for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])

    return best_circles

# EVOLVE-BLOCK-END
