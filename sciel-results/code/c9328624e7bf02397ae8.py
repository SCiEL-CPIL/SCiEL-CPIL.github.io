"""Circle packing in perimeter-4 rectangle: gradient-based softplus penalty optimization
with Delaunay-accelerated polish, multi-phase increasing-beta refinement,
equalization pressure, and SA joint aspect-ratio search."""
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 21
EPS = 1e-9


def _softplus(x, beta=100.0):
    """Numerically stable softplus."""
    xb = beta * x
    return np.where(xb > 0,
                    x + np.log1p(np.exp(-xb)) / beta,
                    np.log1p(np.exp(xb)) / beta)


def _softplus_grad(x, beta=100.0):
    """Gradient of softplus: sigmoid(beta*x)."""
    return 1.0 / (1.0 + np.exp(-beta * x))


def _objective_and_grad(x, w, h, beta=100.0, eq_weight=0.05):
    """
    Softplus-penalized objective with analytical gradients (vectorized).
    x: shape (3N,) = [x0, y0, r0, x1, y1, r1, ...]
    Minimizes: -sum(r) + boundary penalties + pairwise overlap penalties + equalization penalty.
    """
    c = x.reshape(-1, 3)
    xs = c[:, 0]
    ys = c[:, 1]
    rs = c[:, 2]

    obj = -np.sum(rs)
    grad = np.zeros_like(x)

    # Equalization pressure: penalize variance of radii to push toward symmetric packing
    if eq_weight > 0:
        r_mean = np.mean(rs)
        r_var = np.mean((rs - r_mean) ** 2)
        obj += eq_weight * r_var
        # d/dr_i of eq_weight * mean((rs - r_mean)^2) = eq_weight * 2*(rs_i - r_mean)/N
        grad[2::3] += eq_weight * 2.0 * (rs - r_mean) / N

    # Boundary penalties
    v1 = rs - xs          # x - r < 0  ->  v1 > 0 means violation
    obj += np.sum(_softplus(v1, beta))
    g1 = _softplus_grad(v1, beta)
    grad[0::3] -= g1
    grad[2::3] += g1

    v2 = xs + rs - w      # x + r > w
    obj += np.sum(_softplus(v2, beta))
    g2 = _softplus_grad(v2, beta)
    grad[0::3] += g2
    grad[2::3] += g2

    v3 = rs - ys          # y - r < 0
    obj += np.sum(_softplus(v3, beta))
    g3 = _softplus_grad(v3, beta)
    grad[1::3] -= g3
    grad[2::3] += g3

    v4 = ys + rs - h      # y + r > h
    obj += np.sum(_softplus(v4, beta))
    g4 = _softplus_grad(v4, beta)
    grad[1::3] += g4
    grad[2::3] += g4

    v5 = -rs              # r < 0
    obj += np.sum(_softplus(v5, beta))
    g5 = _softplus_grad(v5, beta)
    grad[2::3] -= g5

    # Pairwise overlap penalties (vectorized per i)
    for i in range(N):
        dx = xs[i] - xs[i + 1:]
        dy = ys[i] - ys[i + 1:]
        d = np.sqrt(dx * dx + dy * dy + 1e-12)
        v = rs[i] + rs[i + 1:] - d
        obj += np.sum(_softplus(v, beta))
        g = _softplus_grad(v, beta)

        grad[3 * i]     += -np.sum(g * dx / d)
        grad[3 * i + 1] += -np.sum(g * dy / d)
        grad[3 * i + 2] += np.sum(g)

        idx = np.arange(i + 1, N)
        grad[3 * idx]     += g * dx / d
        grad[3 * idx + 1] += g * dy / d
        grad[3 * idx + 2] += g

    # Gradient of -sum(r)
    grad[2::3] += -1.0

    return obj, grad


def _strictly_valid(circles, w, h, tol=1e-8):
    """Full-pair strict validity check. circles: (N,3)."""
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


def _delaunay_neighbors(centers):
    """Return list of neighbor-index lists via Delaunay; fallback to all pairs."""
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


def _fast_valid(circles, w, h, neighbors, tol=1e-9):
    """Fast validity using precomputed neighbor list (approximate, for polish steps)."""
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


def _hex_init(w, h, seed):
    """Hexagonal lattice init over several row configurations; returns best valid (N,3)."""
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
        # Fallback grid
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
    rng = random.Random(seed)
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


def _optimize_single(circles0, w, h, beta=100.0, max_iter=300, eq_weight=0.05):
    """Run L-BFGS-B from a starting config with given beta; return circles."""
    x0 = circles0.flatten()
    r_max = min(w, h) / 2.0
    bounds = []
    for _ in range(N):
        bounds.append((0.0, w))
        bounds.append((0.0, h))
        bounds.append((1e-6, r_max))
    try:
        res = minimize(
            lambda p: _objective_and_grad(p, w, h, beta, eq_weight),
            x0,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50},
        )
        return res.x.reshape(-1, 3)
    except Exception:
        return circles0.copy()


def _multi_phase_polish(circles0, w, h, deadline, eq_weight=0.05):
    """
    Multi-phase increasing-beta softplus polish: smooth start -> tight local solve.
    Phases: (beta, max_iter) pairs with increasing beta for tighter convergence.
    """
    circles = circles0.copy()
    phases = [
        (20.0, 80),
        (50.0, 150),
        (120.0, 250),
        (300.0, 350),
        (700.0, 450),
        (1500.0, 500),
    ]
    for beta, iters in phases:
        if time.time() > deadline:
            break
        circles = _optimize_single(circles, w, h, beta=beta, max_iter=iters, eq_weight=eq_weight)
    return circles


def _aspect_refine(circles0, w0, h0, deadline, seed=42):
    """
    Targeted aspect-ratio refinement: try small perturbations of w around w0,
    re-optimize circles for each, keep best. Returns (circles, w, h).
    """
    best_circles = circles0.copy()
    best_val = float(np.sum(circles0[:, 2]))
    best_w, best_h = w0, h0
    rng = np.random.default_rng(seed)

    # Try a set of aspect perturbations
    w_candidates = [w0 + d for d in [-0.15, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.15]]
    # Add some random ones
    for _ in range(4):
        w_candidates.append(np.clip(w0 + rng.normal(0, 0.08), 0.45, 1.55))

    for w_new in w_candidates:
        if time.time() > deadline:
            break
        h_new = 2.0 - w_new
        if h_new < 0.45 or w_new < 0.45:
            continue

        # Scale circles to new aspect
        scale_x = w_new / w0
        scale_y = h_new / h0
        circles_scaled = circles0.copy()
        circles_scaled[:, 0] *= scale_x
        circles_scaled[:, 1] *= scale_y
        # Uniformly scale radii by min factor to maintain validity
        scale_r = min(scale_x, scale_y) * 0.95
        circles_scaled[:, 2] *= scale_r

        # Quick polish
        circles_opt = _multi_phase_polish(circles_scaled, w_new, h_new, deadline, eq_weight=0.03)
        circles_opt = _greedy_radius_polish(circles_opt, w_new, h_new, deadline)

        if _strictly_valid(circles_opt, w_new, h_new):
            val = float(np.sum(circles_opt[:, 2]))
            if val > best_val:
                best_val = val
                best_circles = circles_opt.copy()
                best_w, best_h = w_new, h_new

    return best_circles, best_w, best_h


def _greedy_radius_polish(circles, w, h, deadline):
    """Greedy radius-increase polish using Delaunay fast checks, then strict verify."""
    circles = circles.copy()
    centers = circles[:, :2]
    neighbors = _delaunay_neighbors(centers)
    for _ in range(80):
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
        # Refresh neighbor list occasionally
        if _ % 10 == 0:
            centers = circles[:, :2]
            neighbors = _delaunay_neighbors(centers)
    # Final strict verification; if invalid, roll back radii
    if not _strictly_valid(circles, w, h):
        for i in range(N):
            lo, hi = 1e-6, circles[i, 2]
            for _ in range(25):
                mid = 0.5 * (lo + hi)
                circles[i, 2] = mid
                if _strictly_valid(circles, w, h):
                    lo = mid
                else:
                    hi = mid
            circles[i, 2] = lo
    return circles


def _contact_graph_polish(circles, w, h, deadline):
    """
    Identify contact graph (pairs at or near minimum distance) and do
    targeted optimization focusing on those tight constraints.
    """
    if time.time() > deadline - 5:
        return circles

    xs = circles[:, 0].copy()
    ys = circles[:, 1].copy()
    rs = circles[:, 2].copy()

    # Build contact graph: pairs where distance <= r_i + r_j + tol
    tol = 1e-4
    contacts = []
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d = np.sqrt(dx * dx + dy * dy)
            if d <= rs[i] + rs[j] + tol:
                contacts.append((i, j))

    if len(contacts) < 3:
        return circles

    # Also add boundary contacts
    boundary_contacts = []
    for i in range(N):
        if xs[i] - rs[i] <= tol:
            boundary_contacts.append((i, 'left'))
        if xs[i] + rs[i] >= w - tol:
            boundary_contacts.append((i, 'right'))
        if ys[i] - rs[i] <= tol:
            boundary_contacts.append((i, 'bottom'))
        if ys[i] + rs[i] >= h - tol:
            boundary_contacts.append((i, 'top'))

    # Use higher beta and more iterations, with reduced equalization
    # to allow the optimizer to focus on the tight constraints
    x0 = circles.flatten()
    r_max = min(w, h) / 2.0
    bounds = []
    for _ in range(N):
        bounds.append((0.0, w))
        bounds.append((0.0, h))
        bounds.append((1e-6, r_max))

    # Try with very high beta to enforce constraints tightly
    for beta in [2000.0, 5000.0]:
        if time.time() > deadline - 3:
            break
        try:
            res = minimize(
                lambda p: _objective_and_grad(p, w, h, beta, 0.005),
                x0,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 800, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 80},
            )
            new_circles = res.x.reshape(-1, 3)
            if _strictly_valid(new_circles, w, h):
                circles = new_circles
                x0 = circles.flatten()
        except Exception:
            pass

    return circles


def _high_precision_polish(circles, w, h, deadline):
    """Final high-precision polish with very high beta and tight tolerances."""
    if time.time() > deadline - 3:
        return circles
    x0 = circles.flatten()
    r_max = min(w, h) / 2.0
    bounds = []
    for _ in range(N):
        bounds.append((0.0, w))
        bounds.append((0.0, h))
        bounds.append((1e-6, r_max))
    try:
        res = minimize(
            lambda p: _objective_and_grad(p, w, h, 2000.0, 0.005),
            x0,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 800, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 80},
        )
        new_circles = res.x.reshape(-1, 3)
        if _strictly_valid(new_circles, w, h):
            return new_circles
    except Exception:
        pass
    return circles


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    best_circles = None
    best_value = -np.inf
    best_w = best_h = None

    def try_update(circles, w, h):
        nonlocal best_circles, best_value, best_w, best_h
        if circles is not None and _strictly_valid(circles, w, h):
            val = float(np.sum(circles[:, 2]))
            if val > best_value:
                best_value = val
                best_circles = circles.copy()
                best_w, best_h = w, h
            return val
        return -np.inf

    # Aspect ratios: w + h = 2, so h = 2 - w
    # Focus on the most promising ranges with denser sampling
    aspect_ratios = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 0.9, 0.8, 1.7, 0.7, 1.8, 0.65, 1.9]

    seed_counter = 0

    for w in aspect_ratios:
        h = 2.0 - w
        if w < 0.4 or h < 0.4:
            continue
        if time.time() > deadline - 20:
            break

        # Multiple diverse initializations per aspect ratio
        inits = [
            _hex_init(w, h, seed_counter),
            _grid_init(w, h, seed_counter + 1000),
            _random_init(w, h, seed_counter + 2000),
        ]
        seed_counter += 50

        for circles0 in inits:
            if time.time() > deadline - 15:
                break

            # Multi-phase polish: smooth -> tight
            circles = _multi_phase_polish(circles0, w, h, deadline - 5, eq_weight=0.05)

            # Greedy radius polish
            circles = _greedy_radius_polish(circles, w, h, deadline - 3)

            val = try_update(circles, w, h)

            # If promising, try targeted aspect refinement
            if val > 0 and time.time() < deadline - 20:
                ar_circles, ar_w, ar_h = _aspect_refine(
                    circles, w, h, deadline - 10, seed=seed_counter + 3000
                )
                if ar_circles is not None:
                    try_update(ar_circles, ar_w, ar_h)

    # Final high-precision polish on the best found
    if best_circles is not None:
        # Contact-graph-aware polish
        if time.time() < deadline - 8:
            cg_circles = _contact_graph_polish(best_circles, best_w, best_h, deadline - 3)
            try_update(cg_circles, best_w, best_h)

        # High-precision polish
        if time.time() < deadline - 5:
            hp_circles = _high_precision_polish(best_circles, best_w, best_h, deadline - 2)
            try_update(hp_circles, best_w, best_h)

        # Final greedy radius polish
        if time.time() < deadline - 2:
            final_circles = _greedy_radius_polish(best_circles, best_w, best_h, deadline)
            try_update(final_circles, best_w, best_h)

    if best_circles is None:
        # Absolute fallback: tiny valid grid
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
