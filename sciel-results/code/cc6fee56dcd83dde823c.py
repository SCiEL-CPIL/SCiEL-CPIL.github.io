"""Circle packing in perimeter-4 rectangle: Vectorized L-BFGS-B + projection polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-7):
    if W <= 0.01 or H <= 0.01:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    # Pairwise check vectorized
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _vectorized_objective(z, W, H, penalty_weight=2000.0):
    """Smooth penalty objective for L-BFGS-B. Vectorized pairwise."""
    x = z[0:N]
    y = z[N:2*N]
    r = z[2*N:3*N]
    r = np.maximum(r, 0.0)  # ensure non-negative during optimization

    # Objective: maximize sum(r) -> minimize -sum(r)
    val = -np.sum(r)

    # Pairwise overlap penalty (vectorized)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    dist = np.maximum(dist, 1e-12)
    min_dist = r[:, None] + r[None, :]
    overlap = np.maximum(0.0, min_dist - dist)
    # Zero out diagonal
    np.fill_diagonal(overlap, 0.0)
    val += penalty_weight * np.sum(overlap * overlap)

    # Wall penalty (smooth)
    wall_pen = 0.0
    wall_pen += np.sum(np.maximum(0.0, r - x) ** 2)
    wall_pen += np.sum(np.maximum(0.0, x + r - W) ** 2)
    wall_pen += np.sum(np.maximum(0.0, r - y) ** 2)
    wall_pen += np.sum(np.maximum(0.0, y + r - H) ** 2)
    val += penalty_weight * wall_pen

    return val


def _lbfgs_optimize(x0, y0, r0, W, H, time_budget, max_iter=200):
    """Run L-BFGS-B optimization."""
    z0 = np.concatenate([x0, y0, r0])
    bounds = []
    for i in range(N):
        bounds.append((0.0, W))  # x
    for i in range(N):
        bounds.append((0.0, H))  # y
    for i in range(N):
        bounds.append((0.0, min(W, H) / 2))  # r

    try:
        res = minimize(
            _vectorized_objective, z0,
            args=(W, H, 2000.0),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 40}
        )
        x = res.x[0:N]
        y = res.x[N:2*N]
        r = np.maximum(res.x[2*N:3*N], 0.0)
        return x, y, r
    except Exception:
        return None


def _projection_polish(x, y, r, W, H, max_rounds=30, time_budget=None):
    """Iterative projection: for each circle, compute max feasible radius,
    then nudge positions to increase total radius."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)

    for rnd in range(max_rounds):
        if time_budget is not None and time.time() > time_budget:
            break
        improved = False
        for i in range(N):
            # Compute max radius for circle i given current positions
            r_max = min(x[i], W - x[i], y[i], H - y[i])
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(r_max, 0.0)

            if r_max > r[i] + 1e-9:
                r[i] = r_max
                improved = True
            else:
                # Try to move circle i to free up space
                # Find the most constraining direction
                min_gap = float('inf')
                best_dx, best_dy = 0.0, 0.0

                # Walls
                gaps = [
                    (x[i] - r[i], -1.0, 0.0),
                    (W - x[i] - r[i], 1.0, 0.0),
                    (y[i] - r[i], 0.0, -1.0),
                    (H - y[i] - r[i], 0.0, 1.0),
                ]
                for gap, ddx, ddy in gaps:
                    if gap < min_gap:
                        min_gap = gap
                        best_dx, best_dy = ddx, ddy

                # Other circles
                for j in range(N):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d < 1e-10:
                        continue
                    gap = d - r[i] - r[j]
                    if gap < min_gap:
                        min_gap = gap
                        best_dx, best_dy = dx / d, dy / d

                if min_gap < 0.1:
                    # Try small moves
                    for step in [0.05, 0.02, 0.01]:
                        nx = x[i] + best_dx * step
                        ny = y[i] + best_dy * step
                        if nx < r[i] or nx > W - r[i] or ny < r[i] or ny > H - r[i]:
                            continue
                        valid = True
                        for j in range(N):
                            if i == j:
                                continue
                            ddx = nx - x[j]
                            ddy = ny - y[j]
                            if ddx * ddx + ddy * ddy < (r[i] + r[j]) ** 2 - 1e-9:
                                valid = False
                                break
                        if valid:
                            x[i] = nx
                            y[i] = ny
                            improved = True
                            break

        if not improved:
            break

    return x, y, r


def _grid_init(W, H):
    """Grid-based initial placement."""
    # Use a 5x5 grid, pick 21 best positions
    cols, rows = 5, 5
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    count = 0
    for row in range(rows):
        for col in range(cols):
            if count >= N:
                break
            x[count] = (col + 0.5) / cols * W
            y[count] = (row + 0.5) / rows * H
            r[count] = min(W / cols, H / rows) * 0.4
            count += 1
    return x, y, r


def _random_init(W, H, seed):
    """Random initial placement with greedy placement."""
    rng = np.random.RandomState(seed)
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    for i in range(N):
        placed = False
        for attempt in range(200):
            ri = rng.uniform(0.02, min(W, H) * 0.2)
            xi = rng.uniform(ri, W - ri)
            yi = rng.uniform(ri, H - ri)
            valid = True
            for j in range(i):
                dx = xi - x[j]
                dy = yi - y[j]
                if dx * dx + dy * dy < (ri + r[j]) ** 2:
                    valid = False
                    break
            if valid:
                x[i] = xi
                y[i] = yi
                r[i] = ri
                placed = True
                break
        if not placed:
            ri = 0.01
            x[i] = rng.uniform(ri, W - ri)
            y[i] = rng.uniform(ri, H - ri)
            r[i] = ri
    return x, y, r


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin

    best_value = 0.0
    best_result = None

    def update_best(x, y, r, W, H):
        nonlocal best_value, best_result
        if _is_valid(x, y, r, W, H):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), W, H)

    # Aspect ratios to explore (W, H=2-W)
    aspect_ratios = [
        1.0, 0.95, 1.05, 0.9, 1.1, 0.85, 1.15, 0.8, 1.2,
        0.75, 1.25, 0.7, 1.3, 0.65, 1.35, 0.6, 1.4,
        0.55, 1.45, 0.5, 1.5, 1.0, 0.88, 1.12, 0.82, 1.18,
    ]

    # Phase 1: L-BFGS-B from grid and random inits
    for idx, W in enumerate(aspect_ratios):
        if time.time() > deadline - 20:
            break
        H = 2.0 - W
        if H < 0.25:
            continue

        # Grid init
        x, y, r = _grid_init(W, H)
        result = _lbfgs_optimize(x, y, r, W, H, deadline - 15, max_iter=150)
        if result is not None:
            x2, y2, r2 = result
            # Projection polish
            x2, y2, r2 = _projection_polish(x2, y2, r2, W, H, max_rounds=20, time_budget=deadline - 10)
            update_best(x2, y2, r2, W, H)

        # Random init (if time permits)
        if time.time() < deadline - 25:
            x, y, r = _random_init(W, H, seed=idx * 7 + 13)
            result = _lbfgs_optimize(x, y, r, W, H, deadline - 15, max_iter=100)
            if result is not None:
                x2, y2, r2 = result
                x2, y2, r2 = _projection_polish(x2, y2, r2, W, H, max_rounds=15, time_budget=deadline - 10)
                update_best(x2, y2, r2, W, H)

    # Phase 2: Polish the best result further
    if best_result is not None and time.time() < deadline - 10:
        x, y, r, W, H = best_result

        for iteration in range(8):
            if time.time() > deadline - 5:
                break
            x, y, r = _projection_polish(x, y, r, W, H, max_rounds=10, time_budget=deadline - 5)
            update_best(x, y, r, W, H)

            # Try a short L-BFGS refinement
            if time.time() < deadline - 8:
                result = _lbfgs_optimize(x, y, r, W, H, deadline - 5, max_iter=80)
                if result is not None:
                    x2, y2, r2 = result
                    x2, y2, r2 = _projection_polish(x2, y2, r2, W, H, max_rounds=10, time_budget=deadline - 5)
                    update_best(x2, y2, r2, W, H)
                    x, y, r = x2, y2, r2

    # Fallback: simple grid
    if best_result is None:
        W, H = 1.0, 1.0
        x, y, r = _grid_init(W, H)
        best_result = (x, y, r, W, H)

    x, y, r, W, H = best_result

    # Safety shrink
    for _ in range(10):
        r_trial = r * (1.0 - 1e-9)
        if _is_valid(x, y, r_trial, W, H):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
