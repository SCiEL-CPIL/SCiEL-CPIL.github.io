"""Circle packing in unit square: pack 26 circles, maximize sum of radii.

Combines the current program's local-search machinery (best-fit, simulated
annealing) with the inspiration's vectorized gradient-based optimizer,
feasibility projection, and radius-growth polish.
"""
import numpy as np
import time
import random

N = 26

# Deadline guard: 140s wall-clock (10s safety margin under 150s)
_START = None


def _elapsed():
    return time.time() - _START


def _remaining():
    return max(0.0, 140.0 - _elapsed())


# ---------------------------------------------------------------------------
# Vectorized soft-penalty objective with analytic gradient (from inspiration)
# ---------------------------------------------------------------------------

def _soft_objective(x, n=N, mu=1000.0):
    """
    x: (3n,) vector: [x0,y0,r0, x1,y1,r1, ...]
    Returns (energy, grad).
    Energy = -sum(r) + mu * (boundary_penalty + overlap_penalty).
    All computations are vectorized.
    """
    n = len(x) // 3
    pts = x.reshape(n, 3)
    xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]

    obj = -np.sum(rs)

    # Boundary violations
    b_left = np.maximum(0.0, rs - xs)
    b_bot = np.maximum(0.0, rs - ys)
    b_right = np.maximum(0.0, rs - (1.0 - xs))
    b_top = np.maximum(0.0, rs - (1.0 - ys))
    bv = b_left + b_bot + b_right + b_top
    bpen = np.sum(bv ** 2)

    # Overlap violations (vectorized)
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = xs[i_idx] - xs[j_idx]
    dy = ys[i_idx] - ys[j_idx]
    dist_sq = dx ** 2 + dy ** 2
    dist = np.sqrt(dist_sq)
    rs_sum = rs[i_idx] + rs[j_idx]
    ov = np.maximum(0.0, rs_sum - dist)
    open_ = np.sum(ov ** 2)

    total_penalty = bpen + open_
    energy = obj + mu * total_penalty

    # Gradient (vectorized)
    grad = np.zeros_like(x)
    grad[2::3] = -1.0

    db_dxi = np.where(b_left > 0, -1.0, 0.0) + np.where(b_right > 0, 1.0, 0.0)
    db_dyi = np.where(b_bot > 0, -1.0, 0.0) + np.where(b_top > 0, 1.0, 0.0)
    db_dri = (np.where(b_left > 0, 1.0, 0.0) + np.where(b_bot > 0, 1.0, 0.0) +
              np.where(b_right > 0, 1.0, 0.0) + np.where(b_top > 0, 1.0, 0.0))

    grad[0::3] += mu * 2 * bv * db_dxi
    grad[1::3] += mu * 2 * bv * db_dyi
    grad[2::3] += mu * 2 * bv * db_dri

    mask = ov > 0
    if np.any(mask):
        i_m = i_idx[mask]
        j_m = j_idx[mask]
        ov_m = ov[mask]
        dist_m = dist[mask]
        safe_dist = np.where(dist_m < 1e-12, 1.0, dist_m)
        ux = dx[mask] / safe_dist
        uy = dy[mask] / safe_dist

        np.add.at(grad, 3 * i_m, -mu * 2 * ov_m * ux)
        np.add.at(grad, 3 * i_m + 1, -mu * 2 * ov_m * uy)
        np.add.at(grad, 3 * i_m + 2, mu * 2 * ov_m)
        np.add.at(grad, 3 * j_m, mu * 2 * ov_m * ux)
        np.add.at(grad, 3 * j_m + 1, mu * 2 * ov_m * uy)
        np.add.at(grad, 3 * j_m + 2, mu * 2 * ov_m)

    return energy, grad


def _is_valid(x, n=N, tol=1e-6):
    """Check if configuration is valid (all circles inside, no overlaps)."""
    pts = x.reshape(n, 3)
    xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]
    if np.any(xs - rs < -tol) or np.any(ys - rs < -tol):
        return False
    if np.any(1.0 - xs - rs < -tol) or np.any(1.0 - ys - rs < -tol):
        return False
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = xs[i_idx] - xs[j_idx]
    dy = ys[i_idx] - ys[j_idx]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    if np.any(rs[i_idx] + rs[j_idx] - dist > tol):
        return False
    return True


def _project_feasible(x, n=N, max_iter=500, tol=1e-10):
    """
    Project a configuration to feasibility by iteratively resolving violations.
    Reduces radii and adjusts positions to eliminate boundary/overlap violations.
    Vectorized overlap resolution.
    """
    pts = x.reshape(n, 3).copy()
    i_idx, j_idx = np.triu_indices(n, k=1)

    for _ in range(max_iter):
        if _remaining() < 1.0:
            break
        xs, ys, rs = pts[:, 0].copy(), pts[:, 1].copy(), pts[:, 2].copy()

        # Fix boundary violations: reduce radius to fit
        d_left = xs
        d_bot = ys
        d_right = 1.0 - xs
        d_top = 1.0 - ys
        min_d = np.minimum(np.minimum(d_left, d_bot), np.minimum(d_right, d_top))
        too_big = rs > min_d
        if np.any(too_big):
            rs[too_big] = min_d[too_big] * (1 - 1e-9)
            pts[:, 2] = rs

        # Fix overlap violations: reduce radii
        xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]
        dx = xs[i_idx] - xs[j_idx]
        dy = ys[i_idx] - ys[j_idx]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        rs_sum = rs[i_idx] + rs[j_idx]
        ov = np.maximum(0.0, rs_sum - dist)

        if not np.any(ov > tol):
            break

        overlap_per_circle = np.zeros(n)
        np.add.at(overlap_per_circle, i_idx, ov)
        np.add.at(overlap_per_circle, j_idx, ov)

        reduction = overlap_per_circle / 2.0 + tol
        rs = np.maximum(0.01, rs - reduction)
        pts[:, 2] = rs

    return pts.ravel()


def _grow_radii(x, n=N, max_iter=200, step=0.001, tol=1e-8):
    """
    Try to grow radii while maintaining feasibility.
    Uses a greedy approach: try to increase each radius slightly,
    check feasibility, and keep the increase if valid.
    """
    pts = x.reshape(n, 3).copy()
    for _ in range(max_iter):
        if _remaining() < 2.0:
            break

        improved = False
        xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]

        for i in range(n):
            d_left = xs[i]
            d_bot = ys[i]
            d_right = 1.0 - xs[i]
            d_top = ys[i]
            max_r_boundary = min(d_left, d_bot, d_right, d_top)

            max_r_overlap = max_r_boundary
            for j in range(n):
                if i == j:
                    continue
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx ** 2 + dy ** 2)
                limit = dist - rs[j]
                if limit < max_r_overlap:
                    max_r_overlap = limit

            new_r = min(max_r_overlap, rs[i] + step)
            if new_r > rs[i] + 1e-12:
                old_r = rs[i]
                rs[i] = new_r
                pts[i, 2] = new_r

                if _is_valid(pts.ravel(), n, tol=tol):
                    improved = True
                else:
                    rs[i] = old_r
                    pts[i, 2] = old_r

        if not improved:
            break

    return pts.ravel()


def _score(x, n=N):
    """Return sum of radii if valid, else -inf."""
    pts = x.reshape(n, 3)
    if not _is_valid(pts.ravel(), n):
        return -np.inf
    return np.sum(pts[:, 2])


# ---------------------------------------------------------------------------
# Local-search machinery (from current program)
# ---------------------------------------------------------------------------

def _max_radius_at_position(x, y, circles, skip_idx):
    """Compute max radius for a circle at (x,y) given other circles."""
    r_max = min(x, 1.0 - x, y, 1.0 - y)
    if r_max <= 0:
        return 0.0
    for j, (xj, yj, rj) in enumerate(circles):
        if j == skip_idx:
            continue
        d = np.sqrt((x - xj) ** 2 + (y - yj) ** 2)
        r_limit = d - rj
        if r_limit < r_max:
            r_max = r_limit
    return max(0.0, r_max)


def _best_fit_search(circles, iterations=100, grid_res=50):
    """For each circle, find best position on a grid to maximize its radius."""
    circles = [list(c) for c in circles]
    n = len(circles)

    for it in range(iterations):
        if _remaining() < 2.0:
            break
        improved = False
        for i in range(n):
            xi, yi, ri = circles[i]
            best_r = ri
            best_x, best_y = xi, yi

            search_radius = max(ri, 0.05)
            for gx in range(grid_res + 1):
                for gy in range(grid_res + 1):
                    nx = xi + (gx / grid_res - 0.5) * 2 * search_radius
                    ny = yi + (gy / grid_res - 0.5) * 2 * search_radius
                    if nx < 0 or nx > 1 or ny < 0 or ny > 1:
                        continue
                    r_new = _max_radius_at_position(nx, ny, circles, i)
                    if r_new > best_r:
                        best_r = r_new
                        best_x, best_y = nx, ny

            if best_r > ri + 1e-10:
                circles[i] = [best_x, best_y, best_r]
                improved = True

        if not improved:
            break

    return circles


def _simulated_annealing(circles, start_temp=0.01, end_temp=0.001, max_iter=2000):
    """Simulated annealing to escape local optima."""
    circles = [list(c) for c in circles]
    n = len(circles)
    current_sum = sum(c[2] for c in circles)
    best_circles = [list(c) for c in circles]
    best_sum = current_sum

    for it in range(max_iter):
        if _remaining() < 2.0:
            break
        temp = start_temp * (end_temp / start_temp) ** (it / max_iter)

        i = random.randint(0, n - 1)
        xi, yi, ri = circles[i]

        dx = random.gauss(0, 0.05)
        dy = random.gauss(0, 0.05)
        new_x = np.clip(xi + dx, 0.01, 0.99)
        new_y = np.clip(yi + dy, 0.01, 0.99)

        new_r = _max_radius_at_position(new_x, new_y, circles, i)
        new_sum = current_sum - ri + new_r

        delta = new_sum - current_sum
        if delta > 0 or random.random() < np.exp(delta / temp):
            circles[i] = [new_x, new_y, new_r]
            current_sum = new_sum
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = [list(c) for c in circles]

    return best_circles


# ---------------------------------------------------------------------------
# Initializations
# ---------------------------------------------------------------------------

def _initial_grid():
    """Create initial grid of circles."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append([x, y, r])
    return cells


def _hex_packing_init(n=N):
    """Hexagonal packing initial guess."""
    row_counts = [5, 5, 5, 5, 4, 2]
    r_est = 0.09
    y_spacing = r_est * np.sqrt(3)
    x_spacing = 2 * r_est

    xs, ys, rs = [], [], []
    for i, count in enumerate(row_counts):
        y = (i + 0.5) * y_spacing
        total_width = (count - 1) * x_spacing
        x0 = 0.5 - total_width / 2.0
        for j in range(count):
            x = x0 + j * x_spacing
            xs.append(x)
            ys.append(y)
            rs.append(r_est)
    return np.array(list(zip(xs, ys, rs))).ravel()


def _initial_random():
    """Random initialization."""
    random.seed()
    circles = []
    for _ in range(N):
        x = random.uniform(0.05, 0.95)
        y = random.uniform(0.05, 0.95)
        r = 0.02
        circles.append([x, y, r])
    return np.array(circles).ravel()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def construct():
    global _START
    _START = time.time()

    best_x = None
    best_sum = -np.inf

    def _update(x):
        nonlocal best_x, best_sum
        if x is None:
            return
        x = np.asarray(x).ravel()
        if x.size != 3 * N:
            return
        if not _is_valid(x, N):
            # Try to project to feasibility
            x = _project_feasible(x, N)
            if not _is_valid(x, N):
                return
        s = float(np.sum(x.reshape(N, 3)[:, 2]))
        if s > best_sum:
            best_sum = s
            best_x = x.copy()

    # --- Phase 1: gradient-based optimization from multiple initializations ---
    inits = []
    inits.append(_initial_grid())
    inits.append(_hex_packing_init())
    for _ in range(4):
        inits.append(_initial_random())

    for x0 in inits:
        if _remaining() < 25.0:
            break
        # Quick gradient-based optimization
        x = np.asarray(x0).ravel().copy()
        n = N
        lr = 0.01
        for it in range(300):
            if _remaining() < 10.0:
                break
            energy, grad = _soft_objective(x, n, mu=1000.0)
            # Simple gradient descent with decreasing step
            step = lr / (1.0 + 0.05 * it)
            x_new = x - step * grad
            # Clip positions to [0,1] and radii to [0, 0.5]
            x_new[0::3] = np.clip(x_new[0::3], 0.0, 1.0)
            x_new[1::3] = np.clip(x_new[1::3], 0.0, 1.0)
            x_new[2::3] = np.clip(x_new[2::3], 0.01, 0.5)
            x = x_new
        # Project to feasibility and update
        x = _project_feasible(x, N)
        _update(x)

    # --- Phase 2: local search (best-fit) on the best so far ---
    if best_x is not None and _remaining() > 25.0:
        circles = best_x.reshape(N, 3).tolist()
        circles = _best_fit_search(circles, iterations=30, grid_res=40)
        _update(np.array(circles).ravel())

    # --- Phase 3: simulated annealing to escape local optima ---
    if best_x is not None and _remaining() > 20.0:
        circles = best_x.reshape(N, 3).tolist()
        sa_circles = _simulated_annealing(circles, start_temp=0.01, end_temp=0.001, max_iter=1500)
        _update(np.array(sa_circles).ravel())

    # --- Phase 4: radius growth polish ---
    if best_x is not None and _remaining() > 15.0:
        x = _grow_radii(best_x, N, max_iter=150, step=0.001, tol=1e-8)
        _update(x)

    # --- Phase 5: high-precision best-fit polish ---
    if best_x is not None and _remaining() > 10.0:
        circles = best_x.reshape(N, 3).tolist()
        circles = _best_fit_search(circles, iterations=20, grid_res=60)
        _update(np.array(circles).ravel())

    # --- Phase 6: final radius growth ---
    if best_x is not None and _remaining() > 5.0:
        x = _grow_radii(best_x, N, max_iter=100, step=0.0005, tol=1e-9)
        _update(x)

    # Fallback to grid if nothing valid found
    if best_x is None:
        best_x = np.array(_initial_grid()).ravel()
        best_x = _project_feasible(best_x, N)

    if not _is_valid(best_x, N):
        best_x = np.array(_initial_grid()).ravel()
        best_x = _project_feasible(best_x, N)

    return best_x.reshape(N, 3)
