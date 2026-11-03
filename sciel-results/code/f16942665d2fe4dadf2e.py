"""Circle packing in perimeter-4 rectangle: contact-graph-aware optimization
with homotopy beta schedule, multi-aspect-ratio search, and iterative polish."""
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 21
EPS = 1e-9


def _softplus(x, beta=100.0):
    xb = beta * x
    return np.where(xb > 0,
                    x + np.log1p(np.exp(-xb)) / beta,
                    np.log1p(np.exp(xb)) / beta)


def _softplus_grad(x, beta=100.0):
    return 1.0 / (1.0 + np.exp(-beta * x))


def _objective_and_grad(x, w, h, beta=100.0, pairs=None):
    c = x.reshape(-1, 3)
    xs = c[:, 0]
    ys = c[:, 1]
    rs = c[:, 2]

    obj = -np.sum(rs)
    grad = np.zeros_like(x)

    # Boundary penalties
    v1 = rs - xs
    obj += np.sum(_softplus(v1, beta))
    g1 = _softplus_grad(v1, beta)
    grad[0::3] -= g1
    grad[2::3] += g1

    v2 = xs + rs - w
    obj += np.sum(_softplus(v2, beta))
    g2 = _softplus_grad(v2, beta)
    grad[0::3] += g2
    grad[2::3] += g2

    v3 = rs - ys
    obj += np.sum(_softplus(v3, beta))
    g3 = _softplus_grad(v3, beta)
    grad[1::3] -= g3
    grad[2::3] += g3

    v4 = ys + rs - h
    obj += np.sum(_softplus(v4, beta))
    g4 = _softplus_grad(v4, beta)
    grad[1::3] += g4
    grad[2::3] += g4

    v5 = -rs
    obj += np.sum(_softplus(v5, beta))
    g5 = _softplus_grad(v5, beta)
    grad[2::3] -= g5

    # Pairwise overlap penalties
    if pairs is not None:
        for (i, j) in pairs:
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d = np.sqrt(dx * dx + dy * dy + 1e-12)
            v = rs[i] + rs[j] - d
            obj += _softplus(v, beta)
            g = _softplus_grad(v, beta)
            grad[3 * i] += -g * dx / d
            grad[3 * i + 1] += -g * dy / d
            grad[3 * i + 2] += g
            grad[3 * j] += g * dx / d
            grad[3 * j + 1] += g * dy / d
            grad[3 * j + 2] += g
    else:
        for i in range(N):
            dx = xs[i] - xs[i + 1:]
            dy = ys[i] - ys[i + 1:]
            d = np.sqrt(dx * dx + dy * dy + 1e-12)
            v = rs[i] + rs[i + 1:] - d
            obj += np.sum(_softplus(v, beta))
            g = _softplus_grad(v, beta)

            grad[3 * i] += -np.sum(g * dx / d)
            grad[3 * i + 1] += -np.sum(g * dy / d)
            grad[3 * i + 2] += np.sum(g)

            idx = np.arange(i + 1, N)
            grad[3 * idx] += g * dx / d
            grad[3 * idx + 1] += g * dy / d
            grad[3 * idx + 2] += g

    grad[2::3] += -1.0
    return obj, grad


def _strictly_valid(circles, w, h, tol=1e-8):
    n = circles.shape[0]
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    if np.any(rs < 1e-9):
        return False
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
    for i in range(n):
        dx = xs[i] - xs[i + 1:]
        dy = ys[i] - ys[i + 1:]
        d2 = dx * dx + dy * dy
        need = rs[i] + rs[i + 1:] - tol
        if np.any(d2 < need * need):
            return False
    return True


def _delaunay_pairs(circles, w, h, margin=0.05):
    """Get relevant circle pairs using Delaunay triangulation plus proximity check."""
    centers = circles[:, :2]
    rs = circles[:, 2]
    try:
        tri = Delaunay(centers)
        pairs_set = set()
        for simplex in tri.simplices:
            for idx in range(3):
                for jdx in range(idx + 1, 3):
                    i, j = simplex[idx], simplex[jdx]
                    if i > j:
                        i, j = j, i
                    pairs_set.add((i, j))
        # Add close pairs not in Delaunay
        for i in range(N):
            for j in range(i + 1, N):
                if (i, j) in pairs_set:
                    continue
                d = np.sqrt((centers[i, 0] - centers[j, 0]) ** 2 + (centers[i, 1] - centers[j, 1]) ** 2)
                if d < rs[i] + rs[j] + margin:
                    pairs_set.add((i, j))
        return list(pairs_set)
    except Exception:
        return [(i, j) for i in range(N) for j in range(i + 1, N)]


def _fast_valid(circles, w, h, neighbors, tol=1e-9):
    n = circles.shape[0]
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    if np.any(rs < 1e-9):
        return False
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
    for i in range(n):
        for j in neighbors[i]:
            if j <= i:
                continue
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d2 = dx * dx + dy * dy
            need = rs[i] + rs[j] - tol
            if d2 < need * need:
                return False
    return True


def _delaunay_neighbors(centers):
    try:
        tri = Delaunay(centers)
        nbrs = [set() for _ in range(N)]
        for i, j, k in tri.simplices:
            nbrs[i].add(j); nbrs[i].add(k)
            nbrs[j].add(i); nbrs[j].add(k)
            nbrs[k].add(i); nbrs[k].add(j)
        return [list(s) for s in nbrs]
    except Exception:
        return [list(range(N)) for _ in range(N)]


def _hex_init(w, h, seed):
    rng = random.Random(seed)
    row_configs = [
        [3, 4, 4, 4, 3, 3],
        [4, 4, 4, 4, 3, 2],
        [2, 4, 4, 4, 4, 3],
        [3, 4, 4, 4, 4, 2],
        [4, 3, 4, 4, 4, 2],
        [2, 3, 4, 4, 4, 4],
        [3, 3, 4, 4, 4, 3],
        [4, 4, 3, 4, 4, 2],
        [2, 4, 3, 4, 4, 4],
        [3, 4, 3, 4, 4, 3],
        [5, 4, 4, 4, 4],
        [4, 5, 4, 4, 4],
        [3, 5, 4, 4, 5],
        [4, 4, 5, 4, 4],
        [2, 5, 4, 5, 5],
        [5, 5, 5, 5, 1],
        [1, 5, 5, 5, 5],
        [3, 4, 5, 4, 5],
        [5, 4, 5, 4, 3],
    ]
    best = None
    best_val = -np.inf
    for row_counts in row_configs:
        if sum(row_counts) != N:
            continue
        n_rows = len(row_counts)
        max_cols = max(row_counts)
        est_r_h = w / (max_cols + 0.5)
        est_r_v = h / (n_rows * np.sqrt(3) / 2 + 1.0)
        r_est = min(est_r_h, est_r_v) * 0.95
        if r_est < 0.01:
            continue
        circles = []
        for row_idx, n_cols in enumerate(row_counts):
            y = (row_idx + 0.5) * r_est * np.sqrt(3)
            x_offset = r_est if row_idx % 2 == 1 else 0.0
            row_width = n_cols * 2 * r_est
            center_x = w / 2
            for col_idx in range(n_cols):
                x = center_x - row_width / 2 + x_offset + (col_idx + 0.5) * 2 * r_est
                circles.append([x, y, r_est])
        circles = np.array(circles)
        if _strictly_valid(circles, w, h):
            val = float(np.sum(circles[:, 2]))
            if val > best_val:
                best_val = val
                best = circles.copy()
    if best is None:
        a = int(np.ceil(np.sqrt(N * w / h)))
        b = int(np.ceil(N / a))
        while a * b < N:
            b += 1
        circles = []
        for i in range(N):
            col = i % a
            row = i // a
            r = min(w / (a + 0.5), h / (b + 0.5)) * 0.9
            x = (col + 0.5) * w / a
            y = (row + 0.5) * h / b
            circles.append([x, y, r])
        best = np.array(circles)
    return best


def _grid_init(w, h, seed):
    a = int(np.ceil(np.sqrt(N * w / h)))
    b = int(np.ceil(N / a))
    while a * b < N:
        b += 1
    circles = []
    for i in range(N):
        col = i % a
        row = i // a
        r = min(w / (a + 0.5), h / (b + 0.5)) * 0.9
        x = (col + 0.5) * w / a
        y = (row + 0.5) * h / b
        circles.append([x, y, r])
    return np.array(circles)


def _random_init(w, h, seed):
    rng = np.random.RandomState(seed)
    r0 = min(w, h) / 2.0 * 0.25
    circles = np.zeros((N, 3))
    circles[:, 0] = rng.uniform(r0, w - r0, N)
    circles[:, 1] = rng.uniform(r0, h - r0, N)
    circles[:, 2] = r0 * rng.uniform(0.8, 1.2, N)
    return circles


def _farthest_point_init(w, h, n_grid=40):
    """Farthest-point sampling on a grid for well-spaced initial centers."""
    xs_grid = np.linspace(0.05 * w, 0.95 * w, n_grid)
    ys_grid = np.linspace(0.05 * h, 0.95 * h, n_grid)
    X, Y = np.meshgrid(xs_grid, ys_grid)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])

    centers = [grid_points[0].copy()]
    min_dists = np.full(len(grid_points), np.inf)

    for _ in range(N - 1):
        for c in centers:
            d2 = (grid_points[:, 0] - c[0]) ** 2 + (grid_points[:, 1] - c[1]) ** 2
            min_dists = np.minimum(min_dists, d2)
        next_idx = np.argmax(min_dists)
        centers.append(grid_points[next_idx].copy())

    centers = np.array(centers)

    min_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((centers[i, 0] - centers[j, 0]) ** 2 + (centers[i, 1] - centers[j, 1]) ** 2)
            min_dist = min(min_dist, d)
    for i in range(N):
        d_boundary = min(centers[i, 0], w - centers[i, 0], centers[i, 1], h - centers[i, 1])
        min_dist = min(min_dist, d_boundary)

    r_est = min_dist * 0.48
    circles = np.column_stack([centers, np.full(N, r_est)])
    return circles


def _optimize_homotopy(circles0, w, h, deadline, beta_schedule=(20, 50, 100, 200, 400), max_iter_per_stage=150):
    """Optimization with homotopy beta schedule for better convergence."""
    x = circles0.flatten()
    r_max = min(w, h) / 2.0
    bounds = []
    for _ in range(N):
        bounds.append((0.0, w))
        bounds.append((0.0, h))
        bounds.append((1e-6, r_max))

    best_circles = None
    best_val = -np.inf

    for beta in beta_schedule:
        if time.time() > deadline - 3:
            break
        pairs = _delaunay_pairs(circles0 if best_circles is None else best_circles, w, h, margin=0.08)

        def obj_fn(p, beta=beta, pairs=pairs):
            return _objective_and_grad(p, w, h, beta, pairs=pairs)

        try:
            res = minimize(
                obj_fn,
                x,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': max_iter_per_stage, 'ftol': 1e-12, 'gtol': 1e-8},
            )
            if time.time() > deadline - 3:
                break
            circles = res.x.reshape(-1, 3)
            if _strictly_valid(circles, w, h):
                val = float(np.sum(circles[:, 2]))
                if val > best_val:
                    best_val = val
                    best_circles = circles.copy()
                x = res.x
        except Exception:
            pass

    if best_circles is not None:
        return best_circles, best_val
    return None, -np.inf


def _polish(circles, w, h, deadline):
    """Iterative radius expansion with Delaunay-based validity checks."""
    circles = circles.copy()
    centers = circles[:, :2]
    neighbors = _delaunay_neighbors(centers)
    for _ in range(100):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            old_r = circles[i, 2]
            new_r = old_r * 1.005
            circles[i, 2] = new_r
            if _fast_valid(circles, w, h, neighbors):
                improved = True
            else:
                circles[i, 2] = old_r
        if not improved:
            break
        if _ % 10 == 0:
            centers = circles[:, :2]
            neighbors = _delaunay_neighbors(centers)
    if not _strictly_valid(circles, w, h):
        for i in range(N):
            lo, hi = 1e-6, circles[i, 2]
            for _ in range(20):
                mid = 0.5 * (lo + hi)
                circles[i, 2] = mid
                if _strictly_valid(circles, w, h):
                    lo = mid
                else:
                    hi = mid
            circles[i, 2] = lo
    return circles


def _tight_polish(circles, w, h, deadline, beta=400.0):
    """High-precision final optimization with full constraints and high beta."""
    x0 = circles.flatten()
    r_max = min(w, h) / 2.0
    bounds = []
    for _ in range(N):
        bounds.append((0.0, w))
        bounds.append((0.0, h))
        bounds.append((1e-6, r_max))
    try:
        res = minimize(
            lambda p: _objective_and_grad(p, w, h, beta),
            x0,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10},
        )
        if time.time() > deadline:
            return circles
        circles2 = res.x.reshape(-1, 3)
        if _strictly_valid(circles2, w, h):
            circles2 = _polish(circles2, w, h, deadline)
            if _strictly_valid(circles2, w, h):
                return circles2
    except Exception:
        pass
    return circles


def _contact_graph_polish(circles, w, h, deadline):
    """Polish using contact graph structure: identify tight contacts and
    perform local moves that maintain feasibility while expanding radii."""
    circles = circles.copy()
    for outer_iter in range(50):
        if time.time() > deadline:
            break
        centers = circles[:, :2]
        neighbors = _delaunay_neighbors(centers)
        
        # Find tight contacts
        tight_pairs = []
        for i in range(N):
            for j in neighbors[i]:
                if j <= i:
                    continue
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                if d < circles[i, 2] + circles[j, 2] + 0.005:
                    tight_pairs.append((i, j))
        
        # Try expanding each circle slightly
        improved = False
        for i in range(N):
            old_r = circles[i, 2]
            # Try small expansion
            for factor in [1.003, 1.002, 1.001]:
                new_r = old_r * factor
                circles[i, 2] = new_r
                if _fast_valid(circles, w, h, neighbors):
                    improved = True
                    break
                circles[i, 2] = old_r
            if circles[i, 2] > old_r:
                # Recompute neighbors after expansion
                centers = circles[:, :2]
                neighbors = _delaunay_neighbors(centers)
        
        if not improved:
            break
    
    if not _strictly_valid(circles, w, h):
        for i in range(N):
            lo, hi = 1e-6, circles[i, 2]
            for _ in range(20):
                mid = 0.5 * (lo + hi)
                circles[i, 2] = mid
                if _strictly_valid(circles, w, h):
                    lo = mid
                else:
                    hi = mid
            circles[i, 2] = lo
    return circles


def construct():
    start_time = time.time()
    deadline = start_time + 145

    best_circles = None
    best_value = -np.inf
    best_w = best_h = None

    # Aspect ratios: w + h = 2, focus on ratios near optimal
    aspect_ratios = [1.0, 1.2, 1.4, 1.6, 1.8, 1.3, 1.5, 1.7,
                     0.8, 0.9, 1.1, 2.0, 0.7, 2.2, 0.65, 2.35, 0.6,
                     1.35, 1.45, 1.55, 1.65, 0.85, 0.95, 1.15]

    seed_counter = 0

    for w in aspect_ratios:
        h = 2.0 - w
        if w < 0.4 or h < 0.4:
            continue
        if time.time() > deadline - 15:
            break

        # Multiple diverse initializations per aspect ratio
        inits = [
            _hex_init(w, h, seed_counter),
            _grid_init(w, h, seed_counter + 1000),
            _random_init(w, h, seed_counter + 2000),
            _farthest_point_init(w, h),
        ]
        seed_counter += 50

        for circles0 in inits:
            if time.time() > deadline - 10:
                break
            # Homotopy optimization
            circles, val = _optimize_homotopy(circles0, w, h, deadline)
            if circles is not None:
                circles = _polish(circles, w, h, deadline)
                if _strictly_valid(circles, w, h):
                    val = float(np.sum(circles[:, 2]))
                    if val > best_value:
                        best_value = val
                        best_circles = circles.copy()
                        best_w, best_h = w, h

    # Second pass: contact-graph polish on best result
    if best_circles is not None and time.time() < deadline - 15:
        polished = _contact_graph_polish(best_circles, best_w, best_h, deadline)
        if _strictly_valid(polished, best_w, best_h):
            val = float(np.sum(polished[:, 2]))
            if val > best_value:
                best_value = val
                best_circles = polished.copy()

    # Third pass: full-pair refinement with high beta
    if best_circles is not None and time.time() < deadline - 10:
        x0 = best_circles.flatten()
        r_max = min(best_w, best_h) / 2.0
        bounds = []
        for _ in range(N):
            bounds.append((0.0, best_w))
            bounds.append((0.0, best_h))
            bounds.append((1e-6, r_max))
        try:
            res = minimize(
                lambda p: _objective_and_grad(p, best_w, best_h, 200.0),
                x0,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-9},
            )
            circles = res.x.reshape(-1, 3)
            if _strictly_valid(circles, best_w, best_h):
                circles = _polish(circles, best_w, best_h, deadline)
                if _strictly_valid(circles, best_w, best_h):
                    val = float(np.sum(circles[:, 2]))
                    if val > best_value:
                        best_value = val
                        best_circles = circles.copy()
        except Exception:
            pass

    # Final tight polish
    if best_circles is not None and time.time() < deadline - 5:
        best_circles = _tight_polish(best_circles, best_w, best_h, deadline, beta=500.0)
        if _strictly_valid(best_circles, best_w, best_h):
            val = float(np.sum(best_circles[:, 2]))
            if val > best_value:
                best_value = val

    if best_circles is None:
        w, h = 1.4, 0.6
        a, b = 7, 3
        circles = np.zeros((N, 3))
        for i in range(N):
            col = i % a
            row = i // a
            r = min(w / a, h / b) * 0.45
            circles[i, 0] = (col + 0.5) * w / a
            circles[i, 1] = (row + 0.5) * h / b
            circles[i, 2] = r
        best_circles = circles

    return best_circles
