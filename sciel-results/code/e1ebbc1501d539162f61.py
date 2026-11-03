"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.

Strategy:
  - Multi-start from hexagonal / grid / random initializations
  - Delaunay-based LP for per-circle radii (exact feasibility on local constraints)
  - Equal-radius SLSQP refinement jointly optimizing aspect ratio and radius
  - Gradient-based polish (L-BFGS-B) with softplus penalties, Delaunay-reduced overlap
  - Final high-precision polish with tighter tolerance
  - All within a 145 s safety deadline
"""
import numpy as np
import time
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay

N = 21


def _softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(x)))


def _sigmoid(x):
    return np.where(x > 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(-x)))


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
    Maximize sum(r) s.t. r_i + r_j <= d_ij (Delaunay edges), r_i <= boundary, r_i >= 0.
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
            row[i] = 1.0
            row[j] = 1.0
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
    """Largest r such that equal-radius packing is feasible."""
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


def _make_hex(rows):
    """Hexagonal layout with given per-row counts, scaled to perimeter-4 rectangle."""
    pts = []
    for k, n in enumerate(rows):
        y = k * np.sqrt(3.0)
        off = 1.0 if k % 2 == 1 else 0.0
        for i in range(n):
            pts.append((i * 2.0 + off, y))
    pts = np.array(pts[:N])
    xs = pts[:, 0]
    ys = pts[:, 1]
    cx = 0.5 * (xs.max() + xs.min())
    cy = 0.5 * (ys.max() + ys.min())
    xs = xs - cx
    ys = ys - cy
    wc = xs.max() - xs.min()
    hc = ys.max() - ys.min()
    wt = wc + 2.0
    ht = hc + 2.0
    s = 2.0 / (wt + ht)
    xs_f = xs * s + 0.5 * (s * wt)
    ys_f = ys * s + 0.5 * (s * ht)
    return xs_f, ys_f, s * wt, s * ht


def _make_grid(rows, cols):
    pts = []
    for j in range(rows):
        for i in range(cols):
            if len(pts) >= N:
                break
            pts.append((i * 2.0, j * 2.0))
    pts = np.array(pts[:N])
    xs = pts[:, 0]
    ys = pts[:, 1]
    cx = 0.5 * (xs.max() + xs.min())
    cy = 0.5 * (ys.max() + ys.min())
    xs = xs - cx
    ys = ys - cy
    wt = (xs.max() - xs.min()) + 2.0
    ht = (ys.max() - ys.min()) + 2.0
    s = 2.0 / (wt + ht)
    xs_f = xs * s + 0.5 * (s * wt)
    ys_f = ys * s + 0.5 * (s * ht)
    return xs_f, ys_f, s * wt, s * ht


def _make_random(w, h, seed):
    rng = np.random.RandomState(seed)
    xs = rng.uniform(0.1, 0.9, N) * w
    ys = rng.uniform(0.1, 0.9, N) * h
    return xs, ys, w, h


def _refine_slsqp(w, h, xs, ys, r, deadline):
    """
    Equal-radius SLSQP refinement.
    Variables: [w, r, x0..x20, y0..y20].  h = 2 - w.
    Maximize r  s.t.  boundary + pairwise non-overlap.
    """
    if time.time() > deadline - 3.0:
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
                       options={'maxiter': 120, 'ftol': 1e-9, 'disp': False})
    except Exception:
        return w, h, r, xs, ys

    if res.success:
        w_r = res.x[0]
        r_r = res.x[1]
        h_r = 2.0 - w_r
        xs_r = res.x[2:2 + N]
        ys_r = res.x[2 + N:2 + 2 * N]
        if w_r > 0 and h_r > 0 and r_r > 0:
            return w_r, h_r, r_r, xs_r, ys_r
    return w, h, r, xs, ys


def _polish_gradient(w, h, xs, ys, rs, deadline, max_iter=500, use_delaunay=True):
    """
    Smooth gradient-based polish using L-BFGS-B with softplus penalties.
    Variables: [w, x0..x20, y0..y20, r0..r20]. h = 2 - w.
    Maximizes sum(r_i) subject to boundary and non-overlap via softplus penalties.
    Uses Delaunay edges for overlap penalty to reduce computation.
    """
    if time.time() > deadline - 5.0:
        return w, h, xs, ys, rs

    v0 = np.zeros(1 + 3 * N)
    v0[0] = w
    v0[1:1 + N] = xs
    v0[1 + N:1 + 2 * N] = ys
    v0[1 + 2 * N:1 + 3 * N] = rs

    # Precompute edge pairs
    if use_delaunay:
        edges = _delaunay_edges(xs, ys)
        if edges is not None and len(edges) > 0:
            idx_i = np.array([e[0] for e in edges])
            idx_j = np.array([e[1] for e in edges])
        else:
            idx_i, idx_j = np.triu_indices(N, k=1)
    else:
        idx_i, idx_j = np.triu_indices(N, k=1)

    def objective_and_grad(v):
        w_val = v[0]
        h_val = 2.0 - w_val
        x = v[1:1 + N]
        y = v[1 + N:1 + 2 * N]
        r = v[1 + 2 * N:1 + 3 * N]

        if w_val < 0.05 or h_val < 0.05:
            return 1e6, np.zeros_like(v)

        obj = -np.sum(r)
        grad = np.zeros_like(v)
        grad[1 + 2 * N:1 + 3 * N] = -1.0

        eps_b = 1e-3
        eps_o = 1e-3

        # Boundary violations
        v_l = r - x
        sp_l = eps_b * np.log1p(np.exp(np.clip(v_l / eps_b, -20, 20)))
        obj += 500.0 * np.sum(sp_l)
        sig_l = 1.0 / (1.0 + np.exp(-np.clip(v_l / eps_b, -20, 20)))
        grad[1:1 + N] += -500.0 * sig_l
        grad[1 + 2 * N:1 + 3 * N] += 500.0 * sig_l

        v_r = x + r - w_val
        sp_r = eps_b * np.log1p(np.exp(np.clip(v_r / eps_b, -20, 20)))
        obj += 500.0 * np.sum(sp_r)
        sig_r = 1.0 / (1.0 + np.exp(-np.clip(v_r / eps_b, -20, 20)))
        grad[0] += 500.0 * np.sum(-sig_r)
        grad[1:1 + N] += 500.0 * sig_r
        grad[1 + 2 * N:1 + 3 * N] += 500.0 * sig_r

        v_b = r - y
        sp_b = eps_b * np.log1p(np.exp(np.clip(v_b / eps_b, -20, 20)))
        obj += 500.0 * np.sum(sp_b)
        sig_b = 1.0 / (1.0 + np.exp(-np.clip(v_b / eps_b, -20, 20)))
        grad[1 + N:1 + 2 * N] += -500.0 * sig_b
        grad[1 + 2 * N:1 + 3 * N] += 500.0 * sig_b

        v_t = y + r - h_val
        sp_t = eps_b * np.log1p(np.exp(np.clip(v_t / eps_b, -20, 20)))
        obj += 500.0 * np.sum(sp_t)
        sig_t = 1.0 / (1.0 + np.exp(-np.clip(v_t / eps_b, -20, 20)))
        grad[0] += 500.0 * np.sum(sig_t)
        grad[1 + N:1 + 2 * N] += 500.0 * sig_t
        grad[1 + 2 * N:1 + 3 * N] += 500.0 * sig_t

        # Overlap penalty (Delaunay edges only)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx * dx + dy * dy
        min_dist = r[idx_i] + r[idx_j]
        dist_smooth = np.sqrt(dist_sq + 1e-10)
        g = min_dist - dist_smooth
        sp_o = eps_o * np.log1p(np.exp(np.clip(g / eps_o, -20, 20)))
        obj += 2000.0 * np.sum(sp_o)
        sig_o = 1.0 / (1.0 + np.exp(-np.clip(g / eps_o, -20, 20)))

        coeff = 2000.0 * sig_o / dist_smooth
        np.add.at(grad, 1 + idx_i, coeff * dx)
        np.add.at(grad, 1 + idx_j, -coeff * dx)
        np.add.at(grad, 1 + N + idx_i, coeff * dy)
        np.add.at(grad, 1 + N + idx_j, -coeff * dy)
        np.add.at(grad, 1 + 2 * N + idx_i, 2000.0 * sig_o)
        np.add.at(grad, 1 + 2 * N + idx_j, 2000.0 * sig_o)

        return obj, grad

    def objective_only(v):
        return objective_and_grad(v)[0]

    bounds = [(0.05, 1.95)] + [(0.0, 2.0)] * (3 * N)

    try:
        res = minimize(objective_only, v0, jac=objective_and_grad, method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-14, 'disp': False})
    except Exception:
        return w, h, xs, ys, rs

    if res.success or res.fun < 1e3:
        w_r = res.x[0]
        h_r = 2.0 - w_r
        xs_r = res.x[1:1 + N]
        ys_r = res.x[1 + N:1 + 2 * N]
        rs_r = np.maximum(0.0, res.x[1 + 2 * N:1 + 3 * N])
        if w_r > 0.05 and h_r > 0.05:
            return w_r, h_r, xs_r, ys_r, rs_r
    return w, h, xs, ys, rs


def _polish_all_pairs(w, h, xs, ys, rs, deadline, max_iter=300):
    """Polish with all pairwise constraints (for final high-precision pass)."""
    if time.time() > deadline - 5.0:
        return w, h, xs, ys, rs
    return _polish_gradient(w, h, xs, ys, rs, deadline, max_iter=max_iter, use_delaunay=False)


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
    return result


def _validity_margin(result, tol=1e-5):
    """Return minimum clearance (positive = valid). Vectorized."""
    xs = result[:, 0]
    ys = result[:, 1]
    rs = result[:, 2]
    if np.any(rs < 1e-6):
        return -1.0
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    bw = max_x - min_x
    bh = max_y - min_y
    perim = 2.0 * (bw + bh)
    if perim > 4.0 + tol:
        return -1.0
    idx_i, idx_j = np.triu_indices(N, k=1)
    dx = xs[idx_i] - xs[idx_j]
    dy = ys[idx_i] - ys[idx_j]
    dist = np.sqrt(dx * dx + dy * dy)
    min_dist = rs[idx_i] + rs[idx_j]
    m = np.min(dist - min_dist)
    return float(m)


def _make_result_from_lp(w, h, xs, ys):
    rs = _solve_lp_radii(xs, ys, w, h)
    result = np.zeros((N, 3))
    result[:, 0] = xs
    result[:, 1] = ys
    result[:, 2] = np.maximum(rs, 0.0)
    _scale_to_perimeter(result)
    return result


def _make_result_equal(w, h, xs, ys, r):
    result = np.zeros((N, 3))
    result[:, 0] = xs
    result[:, 1] = ys
    result[:, 2] = r
    _scale_to_perimeter(result)
    return result


def _make_result_per_circle(w, h, xs, ys, rs):
    result = np.zeros((N, 3))
    result[:, 0] = xs
    result[:, 1] = ys
    result[:, 2] = np.maximum(rs, 0.0)
    _scale_to_perimeter(result)
    return result


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    np.random.seed(42)

    best_result = None
    best_value = -np.inf

    def consider(result):
        nonlocal best_result, best_value
        if result is None:
            return
        if _validity_margin(result) < -1e-5:
            return
        val = float(np.sum(result[:, 2]))
        if val > best_value:
            best_value = val
            best_result = result.copy()

    # ---------- initial configurations ----------
    inits = []
    hex_rows = [[7, 7, 7], [4, 5, 4, 4, 4], [4, 4, 4, 4, 4, 1],
                [3, 4, 4, 4, 4, 2], [5, 5, 5, 3, 3], [6, 7, 8], [7, 7, 7], [5, 6, 5, 5]]
    for rows in hex_rows:
        if sum(rows) < N:
            continue
        xs, ys, w, h = _make_hex(rows)
        inits.append((w, h, xs, ys))
    grid_configs = [(5, 5), (3, 7), (4, 6), (7, 3), (6, 4), (7, 3), (4, 6)]
    for rows, cols in grid_configs:
        xs, ys, w, h = _make_grid(rows, cols)
        inits.append((w, h, xs, ys))
    for seed in range(8):
        w = 1.0 + 0.15 * (seed - 3.5) / 7.0
        h = 2.0 - w
        xs, ys, w, h = _make_random(w, h, seed + 100)
        inits.append((w, h, xs, ys))

    # ---------- process each init ----------
    for idx, (w, h, xs, ys) in enumerate(inits):
        if time.time() > deadline - 20.0:
            break

        # 1) LP-based per-circle radii result
        res_lp = _make_result_from_lp(w, h, xs, ys)
        consider(res_lp)

        # 2) equal-radius seed from LP max feasible r, then SLSQP refine
        r0 = _max_equal_radius(xs, ys, w, h)
        if r0 > 0:
            res_eq0 = _make_result_equal(w, h, xs, ys, r0)
            consider(res_eq0)
        if time.time() < deadline - 12.0:
            w2, h2, r2, xs2, ys2 = _refine_slsqp(w, h, xs, ys, max(r0, 0.02), deadline)
            r2 = _max_equal_radius(xs2, ys2, w2, h2)
            if r2 > 0:
                res_eq1 = _make_result_equal(w2, h2, xs2, ys2, r2)
                consider(res_eq1)

        # 3) Gradient polish on per-circle radii from LP (Delaunay-reduced)
        if time.time() < deadline - 10.0:
            rs_lp = _solve_lp_radii(xs, ys, w, h)
            if np.any(rs_lp > 0):
                w3, h3, xs3, ys3, rs3 = _polish_gradient(w, h, xs, ys, rs_lp, deadline, max_iter=500)
                if np.any(rs3 > 0):
                    res_poly = _make_result_per_circle(w3, h3, xs3, ys3, rs3)
                    consider(res_poly)
                # Second polish pass
                if time.time() < deadline - 7.0:
                    w4, h4, xs4, ys4, rs4 = _polish_gradient(w3, h3, xs3, ys3, rs3, deadline, max_iter=500)
                    if np.any(rs4 > 0):
                        res_poly2 = _make_result_per_circle(w4, h4, xs4, ys4, rs4)
                        consider(res_poly2)

    # ---------- Final high-precision polish on best result ----------
    if best_result is not None and time.time() < deadline - 12.0:
        xs_b = best_result[:, 0].copy()
        ys_b = best_result[:, 1].copy()
        rs_b = best_result[:, 2].copy()
        min_x = np.min(xs_b - rs_b)
        max_x = np.max(xs_b + rs_b)
        min_y = np.min(ys_b - rs_b)
        max_y = np.max(ys_b + rs_b)
        w_b = max_x - min_x
        h_b = max_y - min_y
        xs_b = xs_b - min_x
        ys_b = ys_b - min_y
        # Polish with Delaunay
        w_f, h_f, xs_f, ys_f, rs_f = _polish_gradient(w_b, h_b, xs_b, ys_b, rs_b, deadline, max_iter=500)
        if np.any(rs_f > 0):
            res_final = _make_result_per_circle(w_f, h_f, xs_f, ys_f, rs_f)
            consider(res_final)
        # Polish with all pairs (high precision)
        if time.time() < deadline - 8.0:
            w_f2, h_f2, xs_f2, ys_f2, rs_f2 = _polish_all_pairs(w_f, h_f, xs_f, ys_f, rs_f, deadline, max_iter=300)
            if np.any(rs_f2 > 0):
                res_final2 = _make_result_per_circle(w_f2, h_f2, xs_f2, ys_f2, rs_f2)
                consider(res_final2)
        # Another Delaunay polish
        if time.time() < deadline - 5.0:
            w_f3, h_f3, xs_f3, ys_f3, rs_f3 = _polish_gradient(w_f2, h_f2, xs_f2, ys_f2, rs_f2, deadline, max_iter=500)
            if np.any(rs_f3 > 0):
                res_final3 = _make_result_per_circle(w_f3, h_f3, xs_f3, ys_f3, rs_f3)
                consider(res_final3)

    # ---------- final fallback ----------
    if best_result is None or best_value <= 0:
        w = 1.0
        h = 1.0
        side = 5
        result = np.zeros((N, 3))
        for i in range(N):
            row = i // side
            col = i % side
            result[i, 0] = (col + 0.5) / side * w
            result[i, 1] = (row + 0.5) / side * h
            result[i, 2] = 0.48 / side
        _scale_to_perimeter(result)
        return result

    return best_result
