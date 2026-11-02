"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.

Combined design:
 - Delaunay-based LP radii solver (exact feasibility for fixed centers)
 - Equal-radius SLSQP refinement jointly optimizing aspect ratio w and radius r
 - Variable-radius L-BFGS-B + SLSQP polish with penalty objective
 - Multiple initializations (hex, grid, random) with LP seeding
"""
import numpy as np
import time
import random
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay

N = 21


def _softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(x)))


def _sigmoid(x):
    return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def _delaunay_edges(xs, ys):
    """Return list of (i, j) edges from Delaunay triangulation of centers."""
    points = np.column_stack((xs, ys))
    try:
        tri = Delaunay(points)
    except Exception:
        return None
    edges = set()
    for simplex in tri.simplices:
        edges.add(tuple(sorted((simplex[0], simplex[1]))))
        edges.add(tuple(sorted((simplex[1], simplex[2]))))
        edges.add(tuple(sorted((simplex[0], simplex[2]))))
    return list(edges)


def _solve_lp_radii(xs, ys, w, h):
    """
    Solve LP for per-circle radii given fixed centers and rectangle.
    Maximize sum(r) s.t.  r_i + r_j <= d_ij  (Delaunay edges)
                          r_i <= min(x_i, w-x_i, y_i, h-y_i)
                          r_i >= 0
    Returns array of radii (zeros on failure).
    """
    limits = np.minimum(np.minimum(xs, w - xs), np.minimum(ys, h - ys))
    if np.any(limits < 0):
        return np.zeros(N)

    A_list = []
    b_list = []
    edges = _delaunay_edges(xs, ys)
    if edges is not None:
        for i, j in edges:
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            row = np.zeros(N)
            row[i] = 1
            row[j] = 1
            A_list.append(row)
            b_list.append(dist)

    A_bound = np.eye(N)
    if A_list:
        A_ub = np.vstack([A_list, A_bound])
        b_ub = np.concatenate([b_list, limits])
    else:
        A_ub = A_bound
        b_ub = limits

    c = -np.ones(N)
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
        if res.success:
            return np.maximum(0.0, res.x)
    except Exception:
        pass
    return np.maximum(0.0, limits * 0.9)


def _max_equal_radius(xs, ys, w, h):
    """
    Largest r such that a single-radius packing is feasible:
    r <= min_i (x_i, w-x_i, y_i, h-y_i)
    r <= d_ij / 2 for all Delaunay edges (i, j).
    """
    limits = np.minimum(np.minimum(xs, w - xs), np.minimum(ys, h - ys))
    r = float(np.min(limits))
    if r < 0:
        return 0.0
    edges = _delaunay_edges(xs, ys)
    if edges is not None:
        for i, j in edges:
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            r = min(r, 0.5 * np.sqrt(dx * dx + dy * dy))
    return max(0.0, r)


def _make_hex(rows_counts, w, h):
    """Hexagonal layout with given per-row counts, scaled to fit w x h."""
    pts = []
    for k, n in enumerate(rows_counts):
        y = k * np.sqrt(3.0)
        off = 1.0 if k % 2 == 1 else 0.0
        for i in range(n):
            pts.append((i * 2.0 + off, y))
    if len(pts) > N:
        pts = pts[:N]
    while len(pts) < N:
        # pad with extra points at the end
        last_x = pts[-1][0] + 2.0
        last_y = pts[-1][1]
        pts.append((last_x, last_y))
    pts = np.array(pts[:N])
    xs = pts[:, 0]
    ys = pts[:, 1]
    cx = 0.5 * (xs.max() + xs.min())
    cy = 0.5 * (ys.max() + ys.min())
    xs = xs - cx
    ys = ys - cy
    wc = xs.max() - xs.min()
    hc = ys.max() - ys.min()
    if wc < 1e-9:
        wc = 1.0
    if hc < 1e-9:
        hc = 1.0
    # Scale to fit in w x h with margin
    s_w = w / (wc + 0.5)
    s_h = h / (hc + 0.5)
    s = min(s_w, s_h)
    xs_f = xs * s + 0.5 * w
    ys_f = ys * s + 0.5 * h
    return xs_f, ys_f


def _make_grid(rows, cols, w, h):
    pts = []
    for j in range(rows):
        for i in range(cols):
            if len(pts) >= N:
                break
            pts.append((i * 2.0, j * 2.0))
    while len(pts) < N:
        last_x = pts[-1][0] + 2.0
        last_y = pts[-1][1]
        pts.append((last_x, last_y))
    pts = np.array(pts[:N])
    xs = pts[:, 0]
    ys = pts[:, 1]
    cx = 0.5 * (xs.max() + xs.min())
    cy = 0.5 * (ys.max() + ys.min())
    xs = xs - cx
    ys = ys - cy
    wc = xs.max() - xs.min()
    hc = ys.max() - ys.min()
    if wc < 1e-9:
        wc = 1.0
    if hc < 1e-9:
        hc = 1.0
    s_w = w / (wc + 0.5)
    s_h = h / (hc + 0.5)
    s = min(s_w, s_h)
    xs_f = xs * s + 0.5 * w
    ys_f = ys * s + 0.5 * h
    return xs_f, ys_f


def _make_random(w, h, seed):
    rng = np.random.RandomState(seed)
    xs = rng.uniform(0.1, 0.9, N) * w
    ys = rng.uniform(0.1, 0.9, N) * h
    return xs, ys


def _scale_to_perimeter(result):
    """Scale result about its bounding-box center so perimeter <= 4."""
    xs = result[:, 0]
    ys = result[:, 1]
    rs = result[:, 2]
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    bw = max_x - min_x
    bh = max_y - min_y
    perim = 2.0 * (bw + bh)
    if perim > 4.0 + 1e-9 and perim > 0:
        scale = 4.0 / perim
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
        for i in range(N):
            result[i, 0] = cx + (result[i, 0] - cx) * scale
            result[i, 1] = cy + (result[i, 1] - cy) * scale
            result[i, 2] = result[i, 2] * scale


def _is_valid(params, tol=1e-5):
    """Check if configuration is valid."""
    w = np.clip(params[0], 0.1, 1.9)
    h = 2.0 - w
    circles = params[1:]
    for i in range(N):
        xi = circles[3 * i]
        yi = circles[3 * i + 1]
        ri = circles[3 * i + 2]
        if ri < 1e-6:
            return False
        if xi - ri < -tol or xi + ri > w + tol:
            return False
        if yi - ri < -tol or yi + ri > h + tol:
            return False
    for i in range(N):
        xi = circles[3 * i]
        yi = circles[3 * i + 1]
        ri = circles[3 * i + 2]
        for j in range(i + 1, N):
            xj = circles[3 * j]
            yj = circles[3 * j + 1]
            rj = circles[3 * j + 2]
            dist_sq = (xi - xj) ** 2 + (yi - yj) ** 2
            if dist_sq < (ri + rj - tol) ** 2:
                return False
    return True


def _get_value(params):
    """Return sum of radii if valid, else -inf."""
    if not _is_valid(params):
        return -np.inf
    w = np.clip(params[0], 0.1, 1.9)
    h = 2.0 - w
    circles = params[1:]
    xs = circles[0::3]
    ys = circles[1::3]
    rs = circles[2::3]
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y
    bbox_perim = 2 * (bbox_w + bbox_h)
    if bbox_perim > 4.0 + 1e-5:
        return -np.inf
    return np.sum(rs)


def _objective_and_grad(params):
    """Penalty-based objective with analytic gradient for variable-radius optimization."""
    W_BOUNDARY = 100.0
    W_OVERLAP = 50.0

    w = params[0]
    w_clipped = np.clip(w, 0.3, 1.7)
    h = 2.0 - w_clipped

    circles = params[1:]
    obj_val = 0.0
    grad = np.zeros_like(params)

    sum_r = 0.0
    for i in range(N):
        xi = circles[3 * i]
        yi = circles[3 * i + 1]
        ri = circles[3 * i + 2]
        if ri < 1e-5:
            ri = 1e-5
        sum_r += ri
        obj_val -= ri
        grad[3 * i + 2] -= 1.0

        # Boundary penalties
        v_left = ri - xi
        if v_left > -20:
            sp_grad = _sigmoid(v_left)
            obj_val += W_BOUNDARY * _softplus(v_left)
            grad[3 * i] -= W_BOUNDARY * sp_grad
            grad[3 * i + 2] += W_BOUNDARY * sp_grad

        v_right = xi + ri - w_clipped
        if v_right > -20:
            sp_grad = _sigmoid(v_right)
            obj_val += W_BOUNDARY * _softplus(v_right)
            grad[3 * i] += W_BOUNDARY * sp_grad
            grad[3 * i + 2] += W_BOUNDARY * sp_grad

        v_bot = ri - yi
        if v_bot > -20:
            sp_grad = _sigmoid(v_bot)
            obj_val += W_BOUNDARY * _softplus(v_bot)
            grad[3 * i + 1] -= W_BOUNDARY * sp_grad
            grad[3 * i + 2] += W_BOUNDARY * sp_grad

        v_top = yi + ri - h
        if v_top > -20:
            sp_grad = _sigmoid(v_top)
            obj_val += W_BOUNDARY * _softplus(v_top)
            grad[3 * i + 1] += W_BOUNDARY * sp_grad
            grad[3 * i + 2] += W_BOUNDARY * sp_grad

    # Overlap penalties
    for i in range(N):
        xi = circles[3 * i]
        yi = circles[3 * i + 1]
        ri = circles[3 * i + 2]
        for j in range(i + 1, N):
            xj = circles[3 * j]
            yj = circles[3 * j + 1]
            rj = circles[3 * j + 2]

            dx = xi - xj
            dy = yi - yj
            dist_sq = dx * dx + dy * dy
            min_dist = ri + rj

            if dist_sq < min_dist * min_dist:
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                v = min_dist - dist
                if v > -20:
                    sp_grad = _sigmoid(v)
                    obj_val += W_OVERLAP * _softplus(v)

                    coeff = W_OVERLAP * sp_grad / dist
                    grad[3 * i] -= coeff * dx
                    grad[3 * i + 1] -= coeff * dy
                    grad[3 * i + 2] += W_OVERLAP * sp_grad

                    grad[3 * j] += coeff * dx
                    grad[3 * j + 1] += coeff * dy
                    grad[3 * j + 2] += W_OVERLAP * sp_grad

    # Gradient for w
    grad_w = 0.0
    for i in range(N):
        yi = circles[3 * i + 1]
        ri = circles[3 * i + 2]
        v_top = yi + ri - h
        if v_top > -20:
            grad_w += W_BOUNDARY * _sigmoid(v_top)
    grad[0] = grad_w

    return obj_val, grad


def _refine_equal_radius_slsqp(w, h, xs, ys, r, deadline):
    """
    Equal-radius SLSQP refinement.
    Variables: [w, r, x0..x20, y0..y20].  h = 2 - w.
    Maximize r  s.t.  boundary + pairwise non-overlap.
    Returns (w, h, r, xs, ys) or the input on failure.
    """
    if time.time() > deadline - 5.0:
        return w, h, r, xs, ys

    v0 = np.zeros(2 + 2 * N)
    v0[0] = w
    v0[1] = r
    v0[2:2 + N] = xs
    v0[2 + N:2 + 2 * N] = ys

    def objective(v):
        if v[0] <= 0.2 or (2.0 - v[0]) <= 0.2 or v[1] <= 0:
            return 1e6
        return -v[1]

    def make_cons():
        cons = [
            {'type': 'ineq', 'fun': lambda v: v[0] - 0.2},
            {'type': 'ineq', 'fun': lambda v: 2.0 - v[0] - 0.2},
        ]
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2 + i] - v[1]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[0] - v[2 + i] - v[1]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2 + N + i] - v[1]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: (2.0 - v[0]) - v[2 + N + i] - v[1]})
        for i in range(N):
            for j in range(i + 1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: np.sqrt((v[2 + i] - v[2 + j]) ** 2
                                                       + (v[2 + N + i] - v[2 + N + j]) ** 2) - 2.0 * v[1]
                })
        return cons

    bounds = [(0.2, 1.8), (0.001, 1.0)] + [(0.0, 2.0)] * (2 * N)

    try:
        res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                       constraints=make_cons(),
                       options={'maxiter': 150, 'ftol': 1e-10, 'disp': False})
    except Exception:
        return w, h, r, xs, ys

    if res.success:
        w_r = res.x[0]
        r_r = res.x[1]
        h_r = 2.0 - w_r
        xs_r = res.x[2:2 + N]
        ys_r = res.x[2 + N:2 + 2 * N]
        if w_r > 0.2 and h_r > 0.2 and r_r > 0:
            return w_r, h_r, r_r, xs_r, ys_r
    return w, h, r, xs, ys


def _refine_variable_radius(params, deadline):
    """
    Variable-radius refinement using L-BFGS-B then SLSQP polish.
    params: [w, x0, y0, r0, x1, y1, r1, ...]
    """
    if time.time() > deadline - 5.0:
        return params

    bounds = [(0.3, 1.7)] + [(-1.0, 3.0)] * (2 * N) + [(1e-5, 1.0)] * N

    # Phase 1: L-BFGS-B
    try:
        result = minimize(
            _objective_and_grad,
            params,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 400, 'ftol': 1e-8}
        )
        params = result.x
    except Exception:
        pass

    # Phase 2: SLSQP polish
    if time.time() > deadline - 3.0:
        return params

    try:
        result = minimize(
            _objective_and_grad,
            params,
            method='SLSQP',
            jac=True,
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        params = result.x
    except Exception:
        pass

    return params


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    best_value = -np.inf
    best_params = None

    random.seed(42)
    np.random.seed(42)

    # Generate candidate initial configurations
    candidates = []

    # 1. Hexagonal patterns with various row structures
    hex_rows = [
        [3, 4, 4, 4, 3, 3],   # 6 rows
        [4, 5, 5, 4, 3],      # 5 rows
        [5, 5, 5, 6],         # 4 rows
        [7, 7, 7],            # 3 rows
        [11, 10],             # 2 rows
        [21],                 # 1 row
    ]

    for w_try in [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]:
        h_try = 2.0 - w_try
        for rows in hex_rows:
            if sum(rows) >= N:
                xs, ys = _make_hex(rows, w_try, h_try)
                # LP radii
                rs = _solve_lp_radii(xs, ys, w_try, h_try)
                if np.sum(rs) > 0:
                    params = np.zeros(1 + 3 * N)
                    params[0] = w_try
                    params[1:] = np.concatenate([xs, ys, rs])
                    candidates.append(params.copy())
                # Equal radius
                r_eq = _max_equal_radius(xs, ys, w_try, h_try)
                if r_eq > 0.005:
                    params = np.zeros(1 + 3 * N)
                    params[0] = w_try
                    for i in range(N):
                        params[3 * i + 1] = xs[i]
                        params[3 * i + 2] = ys[i]
                        params[3 * i + 3] = r_eq
                    candidates.append(params.copy())

    # 2. Grid patterns
    grid_configs = [(3, 7), (4, 6), (5, 5), (6, 4), (7, 3)]
    for w_try in [0.8, 0.9, 1.0, 1.1, 1.2]:
        h_try = 2.0 - w_try
        for (rows, cols) in grid_configs:
            if rows * cols >= N:
                xs, ys = _make_grid(rows, cols, w_try, h_try)
                rs = _solve_lp_radii(xs, ys, w_try, h_try)
                if np.sum(rs) > 0:
                    params = np.zeros(1 + 3 * N)
                    params[0] = w_try
                    params[1:] = np.concatenate([xs, ys, rs])
                    candidates.append(params.copy())

    # 3. Random initializations
    for seed in range(20):
        w_try = random.uniform(0.8, 1.2)
        h_try = 2.0 - w_try
        xs, ys = _make_random(w_try, h_try, seed)
        rs = _solve_lp_radii(xs, ys, w_try, h_try)
        if np.sum(rs) > 0:
            params = np.zeros(1 + 3 * N)
            params[0] = w_try
            params[1:] = np.concatenate([xs, ys, rs])
            candidates.append(params.copy())

    # 4. Perturbed hexagonal
    for seed in range(15):
        w_try = random.uniform(0.85, 1.15)
        h_try = 2.0 - w_try
        rows = [3, 4, 4, 4, 3, 3]
        xs, ys = _make_hex(rows, w_try, h_try)
        xs_p = xs + np.random.randn(N) * 0.01
        ys_p = ys + np.random.randn(N) * 0.01
        # Clamp
        xs_p = np.clip(xs_p, 0.05 * w_try, 0.95 * w_try)
        ys_p = np.clip(ys_p, 0.05 * h_try, 0.95 * h_try)
        rs = _solve_lp_radii(xs_p, ys_p, w_try, h_try)
        if np.sum(rs) > 0:
            params = np.zeros(1 + 3 * N)
            params[0] = w_try
            params[1:] = np.concatenate([xs_p, ys_p, rs])
            candidates.append(params.copy())

    # Evaluate candidates
    for idx, init_params in enumerate(candidates):
        if time.time() > deadline - 5:
            break

        val = _get_value(init_params)
        if val > best_value:
            best_value = val
            best_params = init_params.copy()

    # Refine the best candidates
    # Sort by initial value, take top candidates for refinement
    scored_candidates = []
    for idx, init_params in enumerate(candidates):
        val = _get_value(init_params)
        scored_candidates.append((val, idx, init_params))
    scored_candidates.sort(key=lambda x: -x[0])

    # Refine top 10 with variable-radius optimization
    for val, idx, init_params in scored_candidates[:10]:
        if time.time() > deadline - 10:
            break

        refined = _refine_variable_radius(init_params.copy(), deadline)
        r_val = _get_value(refined)
        if r_val > best_value:
            best_value = r_val
            best_params = refined.copy()

    # Also try equal-radius SLSQP on top candidates
    for val, idx, init_params in scored_candidates[:8]:
        if time.time() > deadline - 8:
            break

        w = init_params[0]
        h = 2.0 - w
        xs = init_params[1::3]
        ys = init_params[2::3]
        rs = init_params[3::3]
        r_eq = float(np.min(rs)) if np.all(rs > 0) else 0.01

        if r_eq < 0.005:
            r_eq = _max_equal_radius(xs, ys, w, h)
            if r_eq < 0.005:
                continue

        w2, h2, r2, xs2, ys2 = _refine_equal_radius_slsqp(w, h, xs, ys, r_eq, deadline)

        # Build params from equal-radius result
        params_eq = np.zeros(1 + 3 * N)
        params_eq[0] = w2
        for i in range(N):
            params_eq[3 * i + 1] = xs2[i]
            params_eq[3 * i + 2] = ys2[i]
            params_eq[3 * i + 3] = r2

        eq_val = _get_value(params_eq)
        if eq_val > best_value:
            best_value = eq_val
            best_params = params_eq.copy()

    # Final polish if we have a good solution
    if best_params is not None and time.time() < deadline - 5:
        polished = _refine_variable_radius(best_params.copy(), deadline)
        p_val = _get_value(polished)
        if p_val > best_value:
            best_value = p_val
            best_params = polished.copy()

    # Fallback
    if best_params is None or best_value < 0:
        w = 1.0
        h = 1.0
        result = np.zeros((N, 3))
        for i in range(N):
            row = i // 5
            col = i % 5
            result[i, 0] = (col + 0.5) / 5
            result[i, 1] = (row + 0.5) / 5
            result[i, 2] = 0.48 / 5
        return result

    # Extract final result
    w_final = np.clip(best_params[0], 0.1, 1.9)
    h_final = 2.0 - w_final
    circles = best_params[1:]

    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = circles[3 * i]
        result[i, 1] = circles[3 * i + 1]
        result[i, 2] = circles[3 * i + 2]

    # Ensure non-negative radii
    result[:, 2] = np.maximum(result[:, 2], 0.0)

    # Final verification: ensure bounding box perimeter <= 4
    _scale_to_perimeter(result)

    return result
