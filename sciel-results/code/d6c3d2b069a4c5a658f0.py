"""Circle packing in unit square for N=26 circles.
Combined approach: multi-strategy search + Delaunay-based constraint reduction + smooth L-BFGS."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26

# EVOLVE-BLOCK-START

def _validate_and_score(circles):
    """Check validity and compute score. circles: (N,3) array [x,y,r]."""
    if circles.shape != (N, 3):
        return None
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    if np.any(r <= 0):
        return None
    if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9) or np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
        return None
    i_idx, j_idx = np.triu_indices(N, k=1)
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dist = np.sqrt(dx * dx + dy * dy)
    if np.any(dist < r[i_idx] + r[j_idx] - 1e-9):
        return None
    return float(np.sum(r))


def _get_delaunay_pairs(x, y, fallback_i, fallback_j):
    """Get Delaunay triangulation edges as index pairs. Falls back to all pairs on failure."""
    points = np.column_stack([x, y])
    try:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    if a > b:
                        a, b = b, a
                    edges.add((a, b))
        if not edges:
            return fallback_i, fallback_j
        edge_list = sorted(edges)
        return np.array([e[0] for e in edge_list]), np.array([e[1] for e in edge_list])
    except Exception:
        return fallback_i, fallback_j


def _grid_initial(n=N):
    """Create a grid-based initial configuration."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            cells.append((cx, cy, r))
    return np.array(cells[:n])


def _random_initial(n=N, seed=42):
    """Create a random initial configuration with non-overlapping circles."""
    rng = np.random.RandomState(seed)
    circles = np.zeros((n, 3))
    circles[:, 2] = 0.01 + 0.01 * rng.rand(n)
    circles[:, 0] = circles[:, 2] + (1 - 2 * circles[:, 2]) * rng.rand(n)
    circles[:, 1] = circles[:, 2] + (1 - 2 * circles[:, 2]) * rng.rand(n)
    return circles


def _structured_initial(n=N, seed=0):
    """Create a structured initial configuration based on hexagonal packing."""
    rng = np.random.RandomState(seed)
    circles = np.zeros((n, 3))
    cols = 5
    rows = 6
    dx = 1.0 / (cols + 0.5)
    dy = 1.0 / (rows + 0.5)
    r_base = min(dx, dy) * 0.48
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
            x_offset = 0.5 * dx if row % 2 == 1 else 0
            cx = (col + 0.5 + x_offset) * dx
            cy = (row + 0.5) * dy
            cx += 0.01 * rng.randn()
            cy += 0.01 * rng.randn()
            cx = np.clip(cx, r_base, 1 - r_base)
            cy = np.clip(cy, r_base, 1 - r_base)
            circles[idx, 0] = cx
            circles[idx, 1] = cy
            circles[idx, 2] = r_base
            idx += 1
    return circles


def _hex_dense_initial(n=N, seed=0):
    """Create a dense hexagonal initial configuration."""
    rng = np.random.RandomState(seed)
    circles = np.zeros((n, 3))
    r = 0.08
    dx = 2 * r * 1.01
    dy = 2 * r * 0.87
    idx = 0
    for row in range(10):
        for col in range(10):
            if idx >= n:
                break
            x_offset = dx * 0.5 if row % 2 == 1 else 0
            cx = dx * 0.5 + col * dx + x_offset
            cy = dy * 0.5 + row * dy
            if cx + r > 1 or cy + r > 1:
                continue
            cx += 0.005 * rng.randn()
            cy += 0.005 * rng.randn()
            cx = np.clip(cx, r, 1 - r)
            cy = np.clip(cy, r, 1 - r)
            circles[idx, 0] = cx
            circles[idx, 1] = cy
            circles[idx, 2] = r
            idx += 1
    return circles[:n]


def _voronoi_initial(n=N, seed=0):
    """Voronoi-based initial: place points, compute Voronoi cells, set radius to inradius."""
    rng = np.random.RandomState(seed)
    points = np.column_stack([rng.rand(n), rng.rand(n)])
    # Add boundary points to constrain Voronoi
    boundary = []
    for i in range(20):
        t = i / 20.0
        boundary.append([t, 0.0])
        boundary.append([t, 1.0])
        boundary.append([0.0, t])
        boundary.append([1.0, t])
    all_pts = np.vstack([points, np.array(boundary)])
    try:
        tri = Delaunay(all_pts)
        # Compute inradius for each of the first n points using Voronoi
        from scipy.spatial import Voronoi
        vor = Voronoi(all_pts)
        radii = []
        for i in range(n):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            if -1 in region or len(region) == 0:
                radii.append(0.02)
                continue
            vertices = vor.vertices[region]
            # Find closest point to center among edges
            cx, cy = points[i]
            min_dist = 1.0
            for k in range(len(vertices)):
                v1 = vertices[k]
                v2 = vertices[(k + 1) % len(vertices)]
                # Distance from point to line segment
                ax, ay = v1
                bx, by = v2
                if ax == cx and ay == cy:
                    continue
                dx = bx - ax
                dy = by - ay
                len2 = dx * dx + dy * dy
                if len2 < 1e-12:
                    continue
                t_param = ((cx - ax) * dx + (cy - ay) * dy) / len2
                t_param = np.clip(t_param, 0, 1)
                px = ax + t_param * dx
                py = ay + t_param * dy
                dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                min_dist = min(min_dist, dist)
            # Also check boundary
            min_dist = min(min_dist, cx, 1 - cx, cy, 1 - cy)
            radii.append(min_dist * 0.95)
        circles = np.column_stack([points, np.array(radii)])
    except Exception:
        circles = _hex_dense_initial(n, seed)
    return circles


def _resolve_overlaps(circles, max_iter=500):
    """Iteratively resolve overlaps by pushing circles apart and shrinking radii."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    i_idx, j_idx = np.triu_indices(n, k=1)
    for it in range(max_iter):
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = r[i_idx] + r[j_idx] - dist
        if not np.any(overlap > 1e-9):
            break
        for k in range(len(i_idx)):
            if overlap[k] > 1e-9:
                i = i_idx[k]
                j = j_idx[k]
                if dist[k] < 1e-12:
                    dx_k = 1e-6
                    dy_k = 0
                else:
                    dx_k = dx[k] / dist[k]
                    dy_k = dy[k] / dist[k]
                push = overlap[k] * 0.5
                x[i] -= push * dx_k
                y[i] -= push * dy_k
                x[j] += push * dx_k
                y[j] += push * dy_k
                shrink = overlap[k] * 0.1
                r[i] = max(r[i] - shrink, 0.001)
                r[j] = max(r[j] - shrink, 0.001)
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        max_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, np.maximum(max_r, 0.001))
    return np.column_stack([x, y, r])


def _delaunay_force_optimize(circles, time_limit=20, seed=0):
    """Physics-based force optimization using Delaunay pairs for efficiency."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    start_time = time.time()
    best_score = _validate_and_score(np.column_stack([x, y, r]))
    best_circles = np.column_stack([x, y, r]).copy()
    if best_score is None:
        best_score = 0.0

    i_all, j_all = np.triu_indices(n, k=1)
    d_i, d_j = _get_delaunay_pairs(x, y, i_all, j_all)

    lr = 0.008
    decay = 0.99999
    momentum_x = np.zeros(n)
    momentum_y = np.zeros(n)
    momentum_r = np.zeros(n)
    beta = 0.85

    it = 0
    no_improve = 0
    max_no_improve = 800

    while True:
        if time.time() - start_time > time_limit:
            break

        if it % 25 == 0:
            d_i, d_j = _get_delaunay_pairs(x, y, i_all, j_all)

        dx = x[d_i] - x[d_j]
        dy = y[d_i] - y[d_j]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        overlap = r[d_i] + r[d_j] - dist

        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)

        mask = overlap > 0
        if np.any(mask):
            rep = 150.0 * overlap[mask] / dist[mask]
            gx[d_i[mask]] -= rep * dx[mask] / dist[mask]
            gy[d_i[mask]] -= rep * dy[mask] / dist[mask]
            gx[d_j[mask]] += rep * dx[mask] / dist[mask]
            gy[d_j[mask]] += rep * dy[mask] / dist[mask]
            gr[d_i[mask]] += 120.0
            gr[d_j[mask]] += 120.0

        gx += np.where(x < r, 300.0, 0.0) - np.where(1 - x < r, 300.0, 0.0)
        gy += np.where(y < r, 300.0, 0.0) - np.where(1 - y < r, 300.0, 0.0)

        gr -= 2.0

        momentum_x = beta * momentum_x + (1 - beta) * gx
        momentum_y = beta * momentum_y + (1 - beta) * gy
        momentum_r = beta * momentum_r + (1 - beta) * gr

        x += lr * momentum_x
        y += lr * momentum_y
        r += lr * momentum_r

        x = np.clip(x, 0.001, 0.999)
        y = np.clip(y, 0.001, 0.999)
        r = np.maximum(r, 0.001)
        max_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, np.maximum(max_r, 0.001))

        lr *= decay
        it += 1

        if it % 15 == 0:
            current_circles = np.column_stack([x, y, r])
            score = _validate_and_score(current_circles)
            if score is not None and score > best_score:
                no_improve = 0
                best_score = score
                best_circles = current_circles.copy()
            else:
                no_improve += 1
            if no_improve >= max_no_improve:
                break

    final_circles = np.column_stack([x, y, r])
    final_score = _validate_and_score(final_circles)
    if final_score is not None and final_score > best_score:
        best_score = final_score
        best_circles = final_circles

    return best_circles, best_score


def _delaunay_polish(circles, time_limit=15, seed=0):
    """High-precision polish using Delaunay contact pairs."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    start_time = time.time()
    best_score = _validate_and_score(np.column_stack([x, y, r]))
    best_circles = np.column_stack([x, y, r]).copy()
    if best_score is None:
        best_score = 0.0

    i_all, j_all = np.triu_indices(n, k=1)
    d_i, d_j = _get_delaunay_pairs(x, y, i_all, j_all)

    lr = 0.004
    decay = 0.99998
    it = 0
    no_improve = 0

    while True:
        if time.time() - start_time > time_limit:
            break

        if it % 20 == 0:
            d_i, d_j = _get_delaunay_pairs(x, y, i_all, j_all)

        dx = x[d_i] - x[d_j]
        dy = y[d_i] - y[d_j]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        overlap = r[d_i] + r[d_j] - dist

        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)

        mask = overlap > 0
        if np.any(mask):
            rep = 200.0 * overlap[mask] / dist[mask]
            gx[d_i[mask]] -= rep * dx[mask] / dist[mask]
            gy[d_i[mask]] -= rep * dy[mask] / dist[mask]
            gx[d_j[mask]] += rep * dx[mask] / dist[mask]
            gy[d_j[mask]] += rep * dy[mask] / dist[mask]
            gr[d_i[mask]] += 150.0
            gr[d_j[mask]] += 150.0

        gx += np.where(x < r, 400.0, 0.0) - np.where(1 - x < r, 400.0, 0.0)
        gy += np.where(y < r, 400.0, 0.0) - np.where(1 - y < r, 400.0, 0.0)

        gr -= 1.0

        x += lr * gx
        y += lr * gy
        r += lr * gr

        x = np.clip(x, 0.001, 0.999)
        y = np.clip(y, 0.001, 0.999)
        r = np.maximum(r, 0.001)
        max_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, np.maximum(max_r, 0.001))

        lr *= decay
        it += 1

        if it % 10 == 0:
            current_circles = np.column_stack([x, y, r])
            score = _validate_and_score(current_circles)
            if score is not None and score > best_score:
                no_improve = 0
                best_score = score
                best_circles = current_circles.copy()
            else:
                no_improve += 1
            if no_improve >= 500:
                break

    final_circles = np.column_stack([x, y, r])
    final_score = _validate_and_score(final_circles)
    if final_score is not None and final_score > best_score:
        best_score = final_score
        best_circles = final_circles

    return best_circles, best_score


def _smooth_lbfgs_optimize(circles, time_limit=15, seed=0):
    """Use L-BFGS-B with smooth penalty method (softplus for constraints)."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    start_time = time.time()
    i_idx, j_idx = np.triu_indices(n, k=1)
    params = np.concatenate([x, y, r])

    W = 80.0
    T = 0.01

    def _softplus(z):
        return T * np.log1p(np.exp(np.clip(z / T, -50, 50)))

    def objective(p):
        if time.time() - start_time > time_limit:
            return 1e10
        xx = p[:n]
        yy = p[n:2 * n]
        rr = p[2 * n:3 * n]
        bv = _softplus(rr - xx) + _softplus(rr - (1 - xx)) + \
              _softplus(rr - yy) + _softplus(rr - (1 - yy))
        dx = xx[i_idx] - xx[j_idx]
        dy = yy[i_idx] - yy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy + 1e-12)
        ov = _softplus(rr[i_idx] + rr[j_idx] - dist)
        return -np.sum(rr) + W * (np.sum(bv) + np.sum(ov))

    def gradient(p):
        if time.time() - start_time > time_limit:
            return np.zeros_like(p)
        xx = p[:n]
        yy = p[n:2 * n]
        rr = p[2 * n:3 * n]

        def _sigmoid(z):
            return 1.0 / (1.0 + np.exp(-np.clip(z / T, -50, 50)))

        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)

        s1 = _sigmoid(rr - xx)
        s2 = _sigmoid(rr - (1 - xx))
        s3 = _sigmoid(rr - yy)
        s4 = _sigmoid(rr - (1 - yy))
        gx += W * (-s1 + s2)
        gy += W * (-s3 + s4)
        gr += W * (s1 + s2 + s3 + s4)

        dx = xx[i_idx] - xx[j_idx]
        dy = yy[i_idx] - yy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy + 1e-12)
        overlap = rr[i_idx] + rr[j_idx] - dist
        s = _sigmoid(overlap)

        gx[i_idx] += W * (-s * dx / dist)
        gx[j_idx] += W * (s * dx / dist)
        gy[i_idx] += W * (-s * dy / dist)
        gy[j_idx] += W * (s * dy / dist)
        gr[i_idx] += W * s
        gr[j_idx] += W * s

        gr -= 1.0

        return np.concatenate([gx, gy, gr])

    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.001, 0.5)] * n

    try:
        result = minimize(objective, params, method='L-BFGS-B', jac=gradient,
                          bounds=bounds, options={'maxiter': 800, 'maxfun': 3000,
                                                   'ftol': 1e-12, 'gtol': 1e-10})
    except Exception:
        input_score = _validate_and_score(circles)
        if input_score is None:
            input_score = 0.0
        return circles.copy(), input_score

    if time.time() - start_time > time_limit:
        input_score = _validate_and_score(circles)
        if input_score is None:
            input_score = 0.0
        return circles.copy(), input_score

    new_params = result.x
    new_x = new_params[:n]
    new_y = new_params[n:2 * n]
    new_r = new_params[2 * n:3 * n]
    new_r = np.maximum(new_r, 0.001)
    new_x = np.clip(new_x, 0, 1)
    new_y = np.clip(new_y, 0, 1)
    max_r = np.minimum(np.minimum(new_x, 1 - new_x), np.minimum(new_y, 1 - new_y))
    new_r = np.minimum(new_r, np.maximum(max_r, 0.001))
    new_circles = np.column_stack([new_x, new_y, new_r])
    new_score = _validate_and_score(new_circles)
    input_score = _validate_and_score(circles)
    if input_score is None:
        input_score = 0.0
    if new_score is None:
        new_score = 0.0
    if new_score >= input_score:
        return new_circles, new_score
    else:
        return circles.copy(), input_score


def _simulated_annealing(circles, max_iter=50000, t_init=0.05, t_min=0.001,
                          time_limit=40, seed=0):
    """Simulated annealing with fast scoring."""
    rng = np.random.RandomState(seed)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    start_time = time.time()
    i_idx, j_idx = np.triu_indices(n, k=1)

    def _penalty_score(cx, cy, cr, penalty_w=50.0):
        if np.any(cr <= 0):
            return -1e6
        bv = np.maximum(0, cr - cx) + np.maximum(0, cr - (1 - cx)) + \
              np.maximum(0, cr - cy) + np.maximum(0, cr - (1 - cy))
        dx = cx[i_idx] - cx[j_idx]
        dy = cy[i_idx] - cy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        ov = np.maximum(0, cr[i_idx] + cr[j_idx] - dist)
        return float(np.sum(cr) - penalty_w * (np.sum(bv) + np.sum(ov)))

    def _strict_score(cx, cy, cr):
        if np.any(cr <= 0):
            return None
        if np.any(cx - cr < -1e-9) or np.any(cx + cr > 1 + 1e-9) or np.any(cy - cr < -1e-9) or np.any(cy + cr > 1 + 1e-9):
            return None
        dx = cx[i_idx] - cx[j_idx]
        dy = cy[i_idx] - cy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        if np.any(dist < cr[i_idx] + cr[j_idx] - 1e-9):
            return None
        return float(np.sum(cr))

    best_strict_score = _strict_score(x, y, r)
    best_circles = np.column_stack([x, y, r]).copy()
    if best_strict_score is None:
        best_strict_score = 0.0
    current_penalty_score = _penalty_score(x, y, r)
    t = t_init
    cooling = (t_min / t_init) ** (1.0 / max_iter)

    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        i = rng.randint(n)
        move_type = rng.rand()
        new_x = x.copy()
        new_y = y.copy()
        new_r = r.copy()
        if move_type < 0.45:
            delta = 0.012 * rng.randn(2)
            new_x[i] += delta[0]
            new_y[i] += delta[1]
        elif move_type < 0.75:
            new_r[i] *= (1 + 0.06 * rng.randn())
            new_r[i] = max(0.005, min(new_r[i], 0.5))
        else:
            j = rng.randint(n)
            if j == i:
                j = (i + 1) % n
            delta_r = 0.008 * rng.randn()
            new_r[i] += delta_r
            new_r[j] -= delta_r
            new_r[i] = max(0.005, new_r[i])
            new_r[j] = max(0.005, new_r[j])
        new_penalty = _penalty_score(new_x, new_y, new_r)
        delta = new_penalty - current_penalty_score
        if delta > 0 or rng.rand() < np.exp(delta / max(t, 1e-10)):
            x, y, r = new_x, new_y, new_r
            current_penalty_score = new_penalty
            if it % 50 == 0:
                s = _strict_score(x, y, r)
                if s is not None and s > best_strict_score:
                    best_strict_score = s
                    best_circles = np.column_stack([x, y, r]).copy()
        t *= cooling

    s = _strict_score(x, y, r)
    if s is not None and s > best_strict_score:
        best_strict_score = s
        best_circles = np.column_stack([x, y, r]).copy()
    return best_circles, best_strict_score


def _refine_radii(circles, time_limit=10):
    """Iteratively grow the most-headroom circles."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    start_time = time.time()
    i_idx, j_idx = np.triu_indices(n, k=1)
    best_score = _validate_and_score(np.column_stack([x, y, r]))
    best_circles = np.column_stack([x, y, r]).copy()
    if best_score is None:
        best_score = 0.0

    for it in range(200):
        if time.time() - start_time > time_limit:
            break
        max_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        for i in range(n):
            headroom = max_r[i] - r[i]
            if headroom < 1e-6:
                continue
            can_grow = True
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r[i] + r[j] + headroom - 1e-9:
                    can_grow = False
                    break
            if can_grow:
                r[i] += headroom * 0.5
                x = np.clip(x, 0, 1)
                y = np.clip(y, 0, 1)
                max_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
                r = np.minimum(r, np.maximum(max_r, 0.001))

        current_circles = np.column_stack([x, y, r])
        score = _validate_and_score(current_circles)
        if score is not None and score > best_score:
            best_score = score
            best_circles = current_circles.copy()

    return best_circles, best_score


def _slsqp_refine(circles, time_limit=15):
    """Use SLSQP with explicit constraints for high-precision refinement."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    start_time = time.time()
    i_idx, j_idx = np.triu_indices(n, k=1)
    params = np.concatenate([x, y, r])

    def objective(p):
        if time.time() - start_time > time_limit:
            return 1e10
        return -np.sum(p[2 * n:3 * n])

    def gradient(p):
        g = np.zeros_like(p)
        g[2 * n:3 * n] = -1.0
        return g

    def constraints():
        cons = []
        # Boundary constraints
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[i] - p[2 * n + i]})
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: (1 - p[i]) - p[2 * n + i]})
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[n + i] - p[2 * n + i]})
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: (1 - p[n + i]) - p[2 * n + i]})
        # Pair non-overlap constraints
        for k in range(len(i_idx)):
            i = i_idx[k]
            j = j_idx[k]
            cons.append({'type': 'ineq', 'fun': lambda p, i=i, j=j: 
                         np.sqrt((p[i] - p[j])**2 + (p[n+i] - p[n+j])**2) - p[2*n+i] - p[2*n+j]})
        return cons

    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.001, 0.5)] * n

    try:
        result = minimize(objective, params, method='SLSQP', jac=gradient,
                          constraints=constraints(), bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-10})
    except Exception:
        input_score = _validate_and_score(circles)
        if input_score is None:
            input_score = 0.0
        return circles.copy(), input_score

    if time.time() - start_time > time_limit:
        input_score = _validate_and_score(circles)
        if input_score is None:
            input_score = 0.0
        return circles.copy(), input_score

    new_params = result.x
    new_x = np.clip(new_params[:n], 0, 1)
    new_y = np.clip(new_params[n:2 * n], 0, 1)
    new_r = np.maximum(new_params[2 * n:3 * n], 0.001)
    max_r = np.minimum(np.minimum(new_x, 1 - new_x), np.minimum(new_y, 1 - new_y))
    new_r = np.minimum(new_r, np.maximum(max_r, 0.001))
    new_circles = np.column_stack([new_x, new_y, new_r])
    new_score = _validate_and_score(new_circles)
    input_score = _validate_and_score(circles)
    if input_score is None:
        input_score = 0.0
    if new_score is None:
        new_score = 0.0
    if new_score >= input_score:
        return new_circles, new_score
    else:
        return circles.copy(), input_score


def construct():
    """Construct optimal circle packing in unit square for N=26 circles."""
    start_time = time.time()
    time_budget = 130
    safety_margin = 20

    best_circles = None
    best_score = 0.0

    initializations = [
        ('grid', _grid_initial(N)),
        ('random1', _random_initial(N, seed=1)),
        ('random2', _random_initial(N, seed=2)),
        ('structured1', _structured_initial(N, seed=1)),
        ('structured2', _structured_initial(N, seed=2)),
        ('hex_dense', _hex_dense_initial(N, seed=0)),
        ('voronoi1', _voronoi_initial(N, seed=1)),
        ('voronoi2', _voronoi_initial(N, seed=42)),
    ]

    for name, init_circles in initializations:
        elapsed = time.time() - start_time
        if elapsed > time_budget - safety_margin:
            break

        resolved = _resolve_overlaps(init_circles, max_iter=200)

        remaining = time_budget - (time.time() - start_time) - safety_margin
        if remaining > 10:
            loc_circles, loc_score = _delaunay_force_optimize(
                resolved, time_limit=min(remaining * 0.25, 15), seed=hash(name) % 1000)
            if loc_score > best_score:
                best_score = loc_score
                best_circles = loc_circles

        remaining = time_budget - (time.time() - start_time) - safety_margin
        if remaining > 10:
            input_c = best_circles if best_circles is not None else resolved
            lbfgs_circles, lbfgs_score = _smooth_lbfgs_optimize(
                input_c, time_limit=min(remaining * 0.25, 12), seed=hash(name) % 1000)
            if lbfgs_score > best_score:
                best_score = lbfgs_score
                best_circles = lbfgs_circles

        remaining = time_budget - (time.time() - start_time) - safety_margin
        if remaining > 15:
            input_c = best_circles if best_circles is not None else resolved
            sa_circles, sa_score = _simulated_annealing(
                input_c, max_iter=40000, t_init=0.04, t_min=0.0005,
                time_limit=min(remaining * 0.3, 20), seed=hash(name) % 1000)
            if sa_score > best_score:
                best_score = sa_score
                best_circles = sa_circles

    # SLSQP refine on best
    if best_circles is not None:
        remaining = time_budget - (time.time() - start_time) - safety_margin
        if remaining > 10:
            slsqp_circles, slsqp_score = _slsqp_refine(
                best_circles, time_limit=min(remaining * 0.3, 15))
            if slsqp_score > best_score:
                best_score = slsqp_score
                best_circles = slsqp_circles

    # Radius refinement pass
    if best_circles is not None:
        remaining = time_budget - (time.time() - start_time) - safety_margin
        if remaining > 8:
            refined_circles, refined_score = _refine_radii(
                best_circles, time_limit=min(remaining * 0.3, 10))
            if refined_score > best_score:
                best_score = refined_score
                best_circles = refined_circles

    # Final Delaunay polish
    if best_circles is not None:
        remaining = time_budget - (time.time() - start_time) - safety_margin
        if remaining > 8:
            final_circles, final_score = _delaunay_polish(
                best_circles, time_limit=min(remaining * 0.7, 18), seed=999)
            if final_score > best_score:
                best_score = final_score
                best_circles = final_circles

    # Final smooth L-BFGS polish
    if best_circles is not None:
        remaining = time_budget - (time.time() - start_time) - safety_margin
        if remaining > 8:
            final_circles, final_score = _smooth_lbfgs_optimize(
                best_circles, time_limit=min(remaining * 0.5, 12), seed=12345)
            if final_score > best_score:
                best_score = final_score
                best_circles = final_circles

    # Final SLSQP polish
    if best_circles is not None:
        remaining = time_budget - (time.time() - start_time) - safety_margin
        if remaining > 8:
            final_circles, final_score = _slsqp_refine(
                best_circles, time_limit=min(remaining * 0.5, 12))
            if final_score > best_score:
                best_score = final_score
                best_circles = final_circles

    if best_circles is None:
        return _grid_initial(N)

    final_score = _validate_and_score(best_circles)
    if final_score is None:
        return _grid_initial(N)

    return best_circles

# EVOLVE-BLOCK-END
