"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.

Combined approach:
 - Delaunay-based LP radii solver (exact feasibility, decoupled structure)
 - Joint aspect-ratio + position + radius optimization via smooth penalty (L-BFGS-B)
 - equal-radius SLSQP refinement jointly optimizing aspect ratio w and radius r
 - hexagonal / grid / random initial configurations
 - Smooth gradient-based polish (L-BFGS-B with softplus penalties) for per-circle radii
 - Force-directed relaxation between optimization phases to escape local minima
 - Active-set high-precision polish
"""
import numpy as np
import time
import random
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay

N = 21


def _softplus(x, beta=100.0):
    xb = beta * x
    return np.where(xb > 0,
                    x + np.log1p(np.exp(-xb)) / beta,
                    np.log1p(np.exp(xb)) / beta)


def _softplus_grad(x, beta=100.0):
    xb = beta * x
    return np.where(xb > 0,
                    1.0 / (1.0 + np.exp(-xb)),
                    np.exp(xb) / (1.0 + np.exp(xb)))


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


def _make_hex(rows, w, h):
    """Hexagonal layout of given per-row counts, scaled to fit w x h."""
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


def _make_grid(rows, cols, w, h):
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


def _force_directed_relax(xs, ys, rs, w, h, deadline, n_steps=200, dt=0.01):
    """
    Physics-based force-directed relaxation to escape local minima.
    Uses repulsive forces between overlapping circles and attractive forces
    to pull circles toward the center, while maintaining boundary constraints.
    Returns updated (xs, ys, rs).
    """
    if time.time() > deadline - 2.0:
        return xs, ys, rs

    xs = xs.copy().astype(float)
    ys = ys.copy().astype(float)
    rs = rs.copy().astype(float)

    for step in range(n_steps):
        if time.time() > deadline - 1.0:
            break

        fx = np.zeros(N)
        fy = np.zeros(N)

        # Repulsive forces between overlapping circles
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = rs[i] + rs[j]
                if dist < 1e-10:
                    dx = 0.01
                    dy = 0.01
                    dist = np.sqrt(dx * dx + dy * dy)
                if dist < min_dist:
                    # Overlap: repulsive force proportional to overlap
                    overlap = min_dist - dist
                    force = 10.0 * overlap
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist

        # Boundary forces: push circles away from walls
        for i in range(N):
            if xs[i] - rs[i] < 0:
                fx[i] += 50.0 * (rs[i] - xs[i])
            if xs[i] + rs[i] > w:
                fx[i] -= 50.0 * (xs[i] + rs[i] - w)
            if ys[i] - rs[i] < 0:
                fy[i] += 50.0 * (rs[i] - ys[i])
            if ys[i] + rs[i] > h:
                fy[i] -= 50.0 * (ys[i] + rs[i] - h)

        # Slight attraction toward center to prevent drift
        cx = 0.5 * w
        cy = 0.5 * h
        for i in range(N):
            fx[i] += 0.1 * (cx - xs[i])
            fy[i] += 0.1 * (cy - ys[i])

        # Update positions with velocity damping
        xs += dt * fx
        ys += dt * fy

        # Clamp to boundaries
        for i in range(N):
            xs[i] = np.clip(xs[i], rs[i], w - rs[i])
            ys[i] = np.clip(ys[i], rs[i], h - rs[i])

        # Try to increase radii slightly if there's room
        for i in range(N):
            margin = min(xs[i] - rs[i], w - xs[i] - rs[i], ys[i] - rs[i], h - ys[i] - rs[i])
            if margin > 1e-4:
                rs[i] = min(rs[i] + 0.001, xs[i], w - xs[i], ys[i], h - ys[i])

    return xs, ys, rs


def _refine_slsqp(w, h, xs, ys, r, deadline):
    """
    Equal-radius SLSQP refinement.
    Variables: [w, r, x0..x20, y0..y20].  h = 2 - w.
    Maximize r  s.t.  boundary + pairwise non-overlap.
    Returns (w, h, r, xs, ys) or the input on failure.
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


def _polish_gradient(w, h, xs, ys, rs, deadline, max_iter=500, beta=100.0):
    """
    Smooth gradient-based polish using L-BFGS-B with softplus penalties.
    Variables: [w, x0..x20, y0..y20, r0..r20]. h = 2 - w.
    Maximizes sum(r_i) subject to boundary and non-overlap via softplus penalties.
    Returns (w, h, xs, ys, rs) or input on failure.
    """
    if time.time() > deadline - 5.0:
        return w, h, xs, ys, rs

    # State vector: [w, x0..x20, y0..y20, r0..r20]
    v0 = np.zeros(1 + 3 * N)
    v0[0] = w
    v0[1:1 + N] = xs
    v0[1 + N:1 + 2 * N] = ys
    v0[1 + 2 * N:1 + 3 * N] = rs

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

        # Boundary violations with softplus
        # Left: r_i - x_i
        v_l = r - x
        sp_l = _softplus(v_l, beta)
        obj += np.sum(sp_l)
        sig_l = _softplus_grad(v_l, beta)
        grad[1:1 + N] += -sig_l
        grad[1 + 2 * N:1 + 3 * N] += sig_l

        # Right: x_i + r_i - w
        v_r = x + r - w_val
        sp_r = _softplus(v_r, beta)
        obj += np.sum(sp_r)
        sig_r = _softplus_grad(v_r, beta)
        grad[0] += -np.sum(sig_r)
        grad[1:1 + N] += sig_r
        grad[1 + 2 * N:1 + 3 * N] += sig_r

        # Bottom: r_i - y_i
        v_b = r - y
        sp_b = _softplus(v_b, beta)
        obj += np.sum(sp_b)
        sig_b = _softplus_grad(v_b, beta)
        grad[1 + N:1 + 2 * N] += -sig_b
        grad[1 + 2 * N:1 + 3 * N] += sig_b

        # Top: y_i + r_i - h
        v_t = y + r - h_val
        sp_t = _softplus(v_t, beta)
        obj += np.sum(sp_t)
        sig_t = _softplus_grad(v_t, beta)
        grad[0] += np.sum(sig_t)
        grad[1 + N:1 + 2 * N] += sig_t
        grad[1 + 2 * N:1 + 3 * N] += sig_t

        # Negative radius penalty
        v_neg = -r
        sp_neg = _softplus(v_neg, beta)
        obj += np.sum(sp_neg)
        sig_neg = _softplus_grad(v_neg, beta)
        grad[1 + 2 * N:1 + 3 * N] -= sig_neg

        # Overlap penalty
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx * dx + dy * dy
        min_dist = r[idx_i] + r[idx_j]
        dist_smooth = np.sqrt(dist_sq + 1e-10)
        g = min_dist - dist_smooth
        sp_o = _softplus(g, beta)
        obj += np.sum(sp_o)
        sig_o = _softplus_grad(g, beta)

        coeff = sig_o / dist_smooth
        np.add.at(grad, 1 + idx_i, coeff * dx)
        np.add.at(grad, 1 + idx_j, -coeff * dx)
        np.add.at(grad, 1 + N + idx_i, coeff * dy)
        np.add.at(grad, 1 + N + idx_j, -coeff * dy)
        np.add.at(grad, 1 + 2 * N + idx_i, sig_o)
        np.add.at(grad, 1 + 2 * N + idx_j, sig_o)

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


def _joint_optimize(w, h, xs, ys, rs, deadline, max_iter=800, beta=50.0):
    """
    Joint optimization of aspect ratio, positions, and radii.
    Variables: [t, x0..x20, y0..y20, r0..r20] where w=2t, h=2(1-t).
    Maximizes sum(r) with smooth penalty for violations.
    """
    if time.time() > deadline - 8.0:
        return w, h, xs, ys, rs

    t0 = w / 2.0
    v0 = np.zeros(1 + 3 * N)
    v0[0] = t0
    v0[1:1 + N] = xs
    v0[1 + N:1 + 2 * N] = ys
    v0[1 + 2 * N:1 + 3 * N] = rs

    idx_i, idx_j = np.triu_indices(N, k=1)

    def objective_and_grad(v):
        t = np.clip(v[0], 0.15, 0.85)
        w_val = 2.0 * t
        h_val = 2.0 * (1.0 - t)
        x = v[1:1 + N]
        y = v[1 + N:1 + 2 * N]
        r = v[1 + 2 * N:1 + 3 * N]

        if w_val < 0.1 or h_val < 0.1:
            return 1e6, np.zeros_like(v)

        obj = -np.sum(r)
        grad = np.zeros_like(v)
        grad[1 + 2 * N:1 + 3 * N] = -1.0

        # Boundary penalties
        v1 = r - x
        obj += np.sum(_softplus(v1, beta))
        g1 = _softplus_grad(v1, beta)
        grad[1:1 + N] -= g1
        grad[1 + 2 * N:1 + 3 * N] += g1

        v2 = x + r - w_val
        obj += np.sum(_softplus(v2, beta))
        g2 = _softplus_grad(v2, beta)
        grad[0] += 2.0 * np.sum(-g2)  # dw/dt = 2
        grad[1:1 + N] += g2
        grad[1 + 2 * N:1 + 3 * N] += g2

        v3 = r - y
        obj += np.sum(_softplus(v3, beta))
        g3 = _softplus_grad(v3, beta)
        grad[1 + N:1 + 2 * N] -= g3
        grad[1 + 2 * N:1 + 3 * N] += g3

        v4 = y + r - h_val
        obj += np.sum(_softplus(v4, beta))
        g4 = _softplus_grad(v4, beta)
        grad[0] += 2.0 * np.sum(g4)  # dh/dt = -2, so d/dt softplus(y+r-h) = g4 * (-dh/dt) = g4*2
        grad[1 + N:1 + 2 * N] += g4
        grad[1 + 2 * N:1 + 3 * N] += g4

        # Negative radius
        v5 = -r
        obj += np.sum(_softplus(v5, beta))
        g5 = _softplus_grad(v5, beta)
        grad[1 + 2 * N:1 + 3 * N] -= g5

        # Pairwise overlap
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx * dx + dy * dy
        min_dist = r[idx_i] + r[idx_j]
        dist_smooth = np.sqrt(dist_sq + 1e-10)
        g = min_dist - dist_smooth
        sp_o = _softplus(g, beta)
        obj += np.sum(sp_o)
        sig_o = _softplus_grad(g, beta)

        coeff = sig_o / dist_smooth
        np.add.at(grad, 1 + idx_i, coeff * dx)
        np.add.at(grad, 1 + idx_j, -coeff * dx)
        np.add.at(grad, 1 + N + idx_i, coeff * dy)
        np.add.at(grad, 1 + N + idx_j, -coeff * dy)
        np.add.at(grad, 1 + 2 * N + idx_i, sig_o)
        np.add.at(grad, 1 + 2 * N + idx_j, sig_o)

        return obj, grad

    def objective_only(v):
        return objective_and_grad(v)[0]

    bounds = [(0.15, 0.85)] + [(0.0, 2.0)] * (3 * N)

    try:
        res = minimize(objective_only, v0, jac=objective_and_grad, method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-14, 'disp': False})
    except Exception:
        return w, h, xs, ys, rs

    if res.success or res.fun < 1e3:
        t_r = np.clip(res.x[0], 0.15, 0.85)
        w_r = 2.0 * t_r
        h_r = 2.0 * (1.0 - t_r)
        xs_r = res.x[1:1 + N]
        ys_r = res.x[1 + N:1 + 2 * N]
        rs_r = np.maximum(0.0, res.x[1 + 2 * N:1 + 3 * N])
        if w_r > 0.1 and h_r > 0.1:
            return w_r, h_r, xs_r, ys_r, rs_r
    return w, h, xs, ys, rs


def _active_set_polish(w, h, xs, ys, rs, deadline, max_iter=100):
    """
    High-precision active-set polish: identify active constraints and solve
    a local system to maximize sum(r) while maintaining feasibility.
    """
    if time.time() > deadline - 3.0:
        return w, h, xs, ys, rs

    xs = xs.copy()
    ys = ys.copy()
    rs = rs.copy()

    for iteration in range(max_iter):
        if time.time() > deadline - 2.0:
            break

        tol = 1e-6
        active_left = rs - xs < tol
        active_right = xs + rs - w < tol
        active_bottom = rs - ys < tol
        active_top = ys + rs - h < tol

        idx_i, idx_j = np.triu_indices(N, k=1)
        dx = xs[idx_i] - xs[idx_j]
        dy = ys[idx_i] - ys[idx_j]
        dist = np.sqrt(dx * dx + dy * dy)
        min_dist = rs[idx_i] + rs[idx_j]
        overlap_violation = min_dist - dist
        active_pair_mask = overlap_violation < tol

        if not np.any(active_left) and not np.any(active_right) and \
           not np.any(active_bottom) and not np.any(active_top) and \
           not np.any(active_pair_mask):
            break

        step = 1e-5
        max_increases = np.full(N, step)

        for i in range(N):
            if active_left[i]:
                max_increases[i] = min(max_increases[i], (xs[i] - rs[i]) + tol)
            if active_right[i]:
                max_increases[i] = min(max_increases[i], (w - xs[i] - rs[i]) + tol)
            if active_bottom[i]:
                max_increases[i] = min(max_increases[i], (ys[i] - rs[i]) + tol)
            if active_top[i]:
                max_increases[i] = min(max_increases[i], (h - ys[i] - rs[i]) + tol)

        for k in range(len(idx_i)):
            if active_pair_mask[k]:
                i = idx_i[k]
                j = idx_j[k]
                current_violation = max_increases[i] + max_increases[j]
                if current_violation > 2 * tol:
                    scale = 2 * tol / current_violation
                    max_increases[i] *= scale
                    max_increases[j] *= scale

        for i in range(N):
            inc = max(0.0, min(max_increases[i], step))
            if inc > 1e-10:
                rs[i] += inc

        if np.all(max_increases < 1e-8):
            break

    return w, h, xs, ys, rs


def _make_result_from_lp(w, h, xs, ys):
    """Build a valid (N,3) result using per-circle LP radii, then scale to perimeter <= 4."""
    rs = _solve_lp_radii(xs, ys, w, h)
    result = np.zeros((N, 3))
    result[:, 0] = xs
    result[:, 1] = ys
    result[:, 2] = rs
    result[:, 2] = np.maximum(result[:, 2], 0.0)
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


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    random.seed(42)
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
    for rows in [[7, 7, 7], [4, 5, 4, 4, 4], [4, 4, 4, 4, 4, 1],
                 [3, 4, 4, 4, 4, 2], [5, 5, 5, 3, 3], [6, 7, 8]]:
        if sum(rows) < N:
            continue
        xs, ys, w, h = _make_hex(rows, 1.0, 1.0)
        inits.append((w, h, xs, ys))
    for rows, cols in [(5, 5), (3, 7), (4, 6), (7, 3), (6, 4)]:
        xs, ys, w, h = _make_grid(rows, cols, 1.0, 1.0)
        inits.append((w, h, xs, ys))
    for seed in range(6):
        w = 1.0 + 0.15 * (seed - 2) / 5.0
        h = 2.0 - w
        xs, ys, w, h = _make_random(w, h, seed + 100)
        inits.append((w, h, xs, ys))

    # ---------- process each init ----------
    for idx, (w, h, xs, ys) in enumerate(inits):
        if time.time() > deadline - 15.0:
            break

        # 1) LP-based per-circle radii result
        res_lp = _make_result_from_lp(w, h, xs, ys)
        consider(res_lp)

        # 2) equal-radius seed from LP max feasible r, then SLSQP refine
        r0 = _max_equal_radius(xs, ys, w, h)
        if r0 > 0:
            res_eq0 = _make_result_equal(w, h, xs, ys, r0)
            consider(res_eq0)
        if time.time() < deadline - 10.0:
            w2, h2, r2, xs2, ys2 = _refine_slsqp(w, h, xs, ys, max(r0, 0.02), deadline)
            r2 = _max_equal_radius(xs2, ys2, w2, h2)
            if r2 > 0:
                res_eq1 = _make_result_equal(w2, h2, xs2, ys2, r2)
                consider(res_eq1)

        # 3) Joint optimization (aspect ratio + positions + radii)
        if time.time() < deadline - 12.0:
            rs_lp = _solve_lp_radii(xs, ys, w, h)
            if np.any(rs_lp > 0):
                w3, h3, xs3, ys3, rs3 = _joint_optimize(w, h, xs, ys, rs_lp, deadline, max_iter=800, beta=50.0)
                if np.any(rs3 > 0):
                    res_joint = _make_result_per_circle(w3, h3, xs3, ys3, rs3)
                    consider(res_joint)

                    # Force-directed relaxation on the joint result
                    if time.time() < deadline - 8.0:
                        xs_fd, ys_fd, rs_fd = _force_directed_relax(xs3, ys3, rs3, w3, h3, deadline, n_steps=150, dt=0.005)
                        if np.any(rs_fd > 0):
                            res_fd = _make_result_per_circle(w3, h3, xs_fd, ys_fd, rs_fd)
                            consider(res_fd)
                            # Polish after force-directed
                            if time.time() < deadline - 6.0:
                                w4, h4, xs4, ys4, rs4 = _polish_gradient(w3, h3, xs_fd, ys_fd, rs_fd, deadline, max_iter=400, beta=200.0)
                                if np.any(rs4 > 0):
                                    res_poly_fd = _make_result_per_circle(w4, h4, xs4, ys4, rs4)
                                    consider(res_poly_fd)

                # 4) Gradient polish on per-circle radii from LP
                if time.time() < deadline - 8.0:
                    w5, h5, xs5, ys5, rs5 = _polish_gradient(w, h, xs, ys, rs_lp, deadline, max_iter=500, beta=100.0)
                    if np.any(rs5 > 0):
                        res_poly = _make_result_per_circle(w5, h5, xs5, ys5, rs5)
                        consider(res_poly)
                        # Second polish pass
                        if time.time() < deadline - 5.0:
                            w6, h6, xs6, ys6, rs6 = _polish_gradient(w5, h5, xs5, ys5, rs5, deadline, max_iter=500, beta=200.0)
                            if np.any(rs6 > 0):
                                res_poly2 = _make_result_per_circle(w6, h6, xs6, ys6, rs6)
                                consider(res_poly2)

    # ---------- Final high-precision polish on best result ----------
    if best_result is not None and time.time() < deadline - 10.0:
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

        # Joint optimization on best
        if time.time() < deadline - 8.0:
            w_j, h_j, xs_j, ys_j, rs_j = _joint_optimize(w_b, h_b, xs_b, ys_b, rs_b, deadline, max_iter=600, beta=100.0)
            if np.any(rs_j > 0):
                res_j = _make_result_per_circle(w_j, h_j, xs_j, ys_j, rs_j)
                consider(res_j)

        # Force-directed relaxation
        if time.time() < deadline - 6.0:
            xs_fd, ys_fd, rs_fd = _force_directed_relax(xs_b, ys_b, rs_b, w_b, h_b, deadline, n_steps=200, dt=0.005)
            if np.any(rs_fd > 0):
                res_fd = _make_result_per_circle(w_b, h_b, xs_fd, ys_fd, rs_fd)
                consider(res_fd)

        # Polish
        if time.time() < deadline - 5.0:
            w_f, h_f, xs_f, ys_f, rs_f = _polish_gradient(w_b, h_b, xs_b, ys_b, rs_b, deadline, max_iter=500, beta=200.0)
            if np.any(rs_f > 0):
                res_final = _make_result_per_circle(w_f, h_f, xs_f, ys_f, rs_f)
                consider(res_final)

            # Active-set polish
            if time.time() < deadline - 3.0:
                w_a, h_a, xs_a, ys_a, rs_a = _active_set_polish(w_f, h_f, xs_f, ys_f, rs_f, deadline, max_iter=100)
                if np.any(rs_a > 0):
                    res_active = _make_result_per_circle(w_a, h_a, xs_a, ys_a, rs_a)
                    consider(res_active)

            # Second gradient polish
            if time.time() < deadline - 4.0:
                w_f2, h_f2, xs_f2, ys_f2, rs_f2 = _polish_gradient(w_f, h_f, xs_f, ys_f, rs_f, deadline, max_iter=500, beta=500.0)
                if np.any(rs_f2 > 0):
                    res_final2 = _make_result_per_circle(w_f2, h_f2, xs_f2, ys_f2, rs_f2)
                    consider(res_final2)
                    if time.time() < deadline - 2.0:
                        w_a2, h_a2, xs_a2, ys_a2, rs_a2 = _active_set_polish(w_f2, h_f2, xs_f2, ys_f2, rs_f2, deadline, max_iter=100)
                        if np.any(rs_a2 > 0):
                            res_active2 = _make_result_per_circle(w_a2, h_a2, xs_a2, ys_a2, rs_a2)
                            consider(res_active2)

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
