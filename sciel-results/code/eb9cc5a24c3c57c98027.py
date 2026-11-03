"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.

Strategy:
  - Delaunay-based LP for per-circle radii (exact feasibility on local constraints)
  - LP-dual (shadow price) driven center optimization: use duals from linprog to
    compute forces on centers, then re-solve LP.
  - Homotopy (beta-scheduled) refinement: start with relaxed constraints, sharpen.
  - Multi-start from hexagonal / grid / random / farthest-point initializations
  - Final SLSQP polish on the best configuration found
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
    return np.where(x > 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


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


def _solve_lp_radii(xs, ys, w, h, beta=1.0):
    """
    Solve LP for per-circle radii given fixed centers and rectangle.
    Maximize sum(r) s.t. r_i + r_j <= d_ij (Delaunay edges), r_i <= boundary, r_i >= 0.
    beta scales the pair constraints: r_i + r_j <= beta * d_ij.
    Returns (radii, duals) where duals is a list of (kind, i, j, value).
    """
    limits = np.minimum(np.minimum(xs, w - xs), np.minimum(ys, h - ys))
    if np.any(limits < 0):
        return np.zeros(N), []

    A_list = []
    b_list = []
    edges = _delaunay_edges(xs, ys)
    pair_idx = []
    if edges is not None:
        for i, j in edges:
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            row = np.zeros(N)
            row[i] = 1.0
            row[j] = 1.0
            A_list.append(row)
            b_list.append(beta * dist)
            pair_idx.append((i, j))

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
            radii = np.maximum(0.0, res.x)
            duals = []
            if hasattr(res, 'ineqlin') and res.ineqlin is not None and hasattr(res.ineqlin, 'marginals'):
                marg = res.ineqlin.marginals
                for k, (i, j) in enumerate(pair_idx):
                    if k < len(marg):
                        duals.append(('pair', i, j, abs(marg[k])))
                offset = len(pair_idx)
                for i in range(N):
                    if offset + i < len(marg):
                        duals.append(('bound', i, -1, abs(marg[offset + i])))
            return radii, duals
    except Exception:
        pass
    return np.maximum(0.0, limits * 0.9), []


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


def _make_farthest_point(w, h, seed):
    """Farthest-point sampling initialization for better spread."""
    rng = np.random.RandomState(seed)
    xs = np.zeros(N)
    ys = np.zeros(N)
    xs[0] = rng.uniform(0.2, 0.8) * w
    ys[0] = rng.uniform(0.2, 0.8) * h
    for k in range(1, N):
        cand_x = rng.uniform(0.05, 0.95, 300) * w
        cand_y = rng.uniform(0.05, 0.95, 300) * h
        min_dists = np.full(300, np.inf)
        for j in range(k):
            d = np.sqrt((cand_x - xs[j]) ** 2 + (cand_y - ys[j]) ** 2)
            min_dists = np.minimum(min_dists, d)
        bd = np.minimum(np.minimum(cand_x, w - cand_x), np.minimum(cand_y, h - cand_y))
        min_dists = np.minimum(min_dists, bd)
        best = np.argmax(min_dists)
        xs[k] = cand_x[best]
        ys[k] = cand_y[best]
    return xs, ys, w, h


def _dual_driven_optimize(xs, ys, w, h, max_iters=200, deadline=None, beta_start=0.5, beta_end=1.0):
    """
    LP-dual driven center optimization with homotopy.
    Iteratively:
      1. Solve LP for radii given centers (with current beta), get duals.
      2. Compute forces on centers from duals.
      3. Update centers with adaptive step size.
      4. Gradually increase beta from beta_start to beta_end.
    Returns best (xs, ys, radii, value) found.
    """
    xs = xs.copy().astype(float)
    ys = ys.copy().astype(float)
    best_val = -np.inf
    best_xs, best_ys, best_rs = xs.copy(), ys.copy(), np.zeros(N)

    step = 0.015
    stall = 0
    rng = np.random.RandomState(12345)

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 3.0:
            break

        # Homotopy: linearly interpolate beta
        beta = beta_start + (beta_end - beta_start) * (it / max(1, max_iters))
        beta = min(1.0, beta)

        radii, duals = _solve_lp_radii(xs, ys, w, h, beta=beta)
        val = float(np.sum(radii))

        if val > best_val:
            best_val = val
            best_xs = xs.copy()
            best_ys = ys.copy()
            best_rs = radii.copy()
            stall = 0
        else:
            stall += 1

        if stall > 30:
            step *= 0.7
            xs += rng.normal(0, 0.003, N)
            ys += rng.normal(0, 0.003, N)
            xs = np.clip(xs, 1e-4, w - 1e-4)
            ys = np.clip(ys, 1e-4, h - 1e-4)
            stall = 0
            continue

        if stall > 15 and step > 0.002:
            step *= 0.85

        # Compute forces from duals
        fx = np.zeros(N)
        fy = np.zeros(N)
        for kind, i, j, dv in duals:
            if dv < 1e-10:
                continue
            if kind == 'pair':
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-12:
                    continue
                ux = dx / dist
                uy = dy / dist
                fx[i] += dv * ux
                fy[i] += dv * uy
                fx[j] -= dv * ux
                fy[j] -= dv * uy
            elif kind == 'bound':
                # Boundary dual: push center away from boundary
                if xs[i] < w / 2:
                    fx[i] += dv
                else:
                    fx[i] -= dv
                if ys[i] < h / 2:
                    fy[i] += dv
                else:
                    fy[i] -= dv

        # Normalize and apply
        fmag = np.sqrt(np.sum(fx ** 2) + np.sum(fy ** 2))
        if fmag > 1e-12:
            fx = fx / fmag * step
            fy = fy / fmag * step
            xs += fx
            ys += fy
            xs = np.clip(xs, 1e-4, w - 1e-4)
            ys = np.clip(ys, 1e-4, h - 1e-4)

    # Final LP solve at beta=1.0 for exact radii
    radii_final, _ = _solve_lp_radii(best_xs, best_ys, w, h, beta=1.0)
    val_final = float(np.sum(radii_final))
    if val_final > best_val:
        best_val = val_final
        best_rs = radii_final.copy()

    return best_xs, best_ys, best_rs, best_val


def _check_valid_var(xs, ys, rs, w, h, tol=1e-7):
    """Check validity of variable-radius configuration."""
    for i in range(N):
        if rs[i] < 1e-6:
            return False
        if xs[i] - rs[i] < -tol or xs[i] + rs[i] > w + tol:
            return False
        if ys[i] - rs[i] < -tol or ys[i] + rs[i] > h + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)
            if d < rs[i] + rs[j] - tol:
                return False
    return True


def _bbox_perim(xs, ys, rs):
    """Compute bounding box perimeter."""
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    return 2.0 * ((max_x - min_x) + (max_y - min_y))


def _objective_variable_radius(params, w, h, W=200.0):
    """
    Objective for variable-radius packing: minimize -sum(ri) + penalties.
    params = [x1, y1, r1, x2, y2, r2, ...] (3*N)
    """
    n = N
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]

    obj = -np.sum(rs)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0

    # Boundary penalties
    for i in range(n):
        xi, yi, ri = xs[i], ys[i], rs[i]
        if ri < 1e-8:
            ri = 1e-8
        v = ri - xi
        if v > -20:
            s = _sigmoid(v)
            obj += W * _softplus(v)
            grad[3 * i] -= W * s
            grad[3 * i + 2] += W * s
        v = xi + ri - w
        if v > -20:
            s = _sigmoid(v)
            obj += W * _softplus(v)
            grad[3 * i] += W * s
            grad[3 * i + 2] += W * s
        v = ri - yi
        if v > -20:
            s = _sigmoid(v)
            obj += W * _softplus(v)
            grad[3 * i + 1] -= W * s
            grad[3 * i + 2] += W * s
        v = yi + ri - h
        if v > -20:
            s = _sigmoid(v)
            obj += W * _softplus(v)
            grad[3 * i + 1] += W * s
            grad[3 * i + 2] += W * s

    # Pairwise overlap
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx * dx + dy * dy
            min_d = rs[i] + rs[j]
            if dist_sq < min_d * min_d:
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                v = min_d - dist
                if v > -20:
                    s = _sigmoid(v)
                    obj += W * _softplus(v)
                    coeff = W * s / dist
                    grad[3 * i] -= coeff * dx
                    grad[3 * i + 1] -= coeff * dy
                    grad[3 * i + 2] += W * s
                    grad[3 * j] += coeff * dx
                    grad[3 * j + 1] += coeff * dy
                    grad[3 * j + 2] += W * s

    return obj, grad


def _slsqp_polish(xs, ys, rs, w, h, deadline):
    """SLSQP polish on a given configuration."""
    params = np.concatenate([xs, ys, rs])
    bounds = [(0, w)] * N + [(0, h)] * N + [(1e-4, 0.5)] * N

    try:
        res = minimize(
            lambda p: _objective_variable_radius(p, w, h),
            params,
            jac=True,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-14}
        )
        p_opt = res.x
        xs_opt = p_opt[0::3]
        ys_opt = p_opt[1::3]
        rs_opt = p_opt[2::3]

        if _check_valid_var(xs_opt, ys_opt, rs_opt, w, h):
            return xs_opt, ys_opt, rs_opt, float(np.sum(rs_opt))
    except Exception:
        pass

    return xs, ys, rs, float(np.sum(rs))


def construct():
    start_time = time.time()
    deadline = start_time + 143.0  # safety margin before 150s kill

    np.random.seed(42)

    best_value = -np.inf
    best_result = None

    # Aspect ratios to try
    aspect_ratios = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1]

    for ar in aspect_ratios:
        if time.time() > deadline - 15:
            break

        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 - w

        # Generate initializations
        inits = []

        # Hexagonal layouts
        hex_configs = [[7, 7, 7], [6, 7, 8], [7, 8, 6], [5, 8, 8], [8, 7, 6]]
        for rows in hex_configs:
            if sum(rows) >= N:
                xs, ys, ww, hh = _make_hex(rows)
                # Scale to target w, h
                xs = xs * (w / ww)
                ys = ys * (h / hh)
                inits.append((xs.copy(), ys.copy()))

        # Grid layouts
        for rows, cols in [(3, 7), (4, 6), (5, 5), (7, 3), (6, 4)]:
            if rows * cols >= N:
                xs, ys, ww, hh = _make_grid(rows, cols)
                xs = xs * (w / ww)
                ys = ys * (h / hh)
                inits.append((xs.copy(), ys.copy()))

        # Random and farthest-point
        for seed in [1, 7, 42, 100, 200]:
            xs, ys, ww, hh = _make_random(w, h, seed)
            inits.append((xs.copy(), ys.copy()))
        for seed in [3, 11, 55]:
            xs, ys, ww, hh = _make_farthest_point(w, h, seed)
            inits.append((xs.copy(), ys.copy()))

        for xs_init, ys_init in inits:
            if time.time() > deadline - 10:
                break

            # LP-dual driven optimization with homotopy
            xs_opt, ys_opt, rs_opt, val = _dual_driven_optimize(
                xs_init, ys_init, w, h,
                max_iters=150, deadline=deadline,
                beta_start=0.5, beta_end=1.0
            )

            if val > best_value:
                best_value = val
                best_result = np.column_stack([xs_opt, ys_opt, rs_opt])

        # SLSQP polish on the best found for this aspect ratio
        if best_result is not None and time.time() < deadline - 8:
            # Find the best result for this specific w, h
            # We'll polish the global best
            xs_b = best_result[:, 0].copy()
            ys_b = best_result[:, 1].copy()
            rs_b = best_result[:, 2].copy()

            # Try to re-scale best_result to this w,h if it came from different aspect
            # Actually, let's just polish with current w,h
            xs_p, ys_p, rs_p, val_p = _slsqp_polish(xs_b, ys_b, rs_b, w, h, deadline)
            if val_p > best_value:
                best_value = val_p
                best_result = np.column_stack([xs_p, ys_p, rs_p])

    # Final global SLSQP polish
    if best_result is not None and time.time() < deadline - 5:
        # Determine the effective w, h from the bounding box
        xs_f = best_result[:, 0]
        ys_f = best_result[:, 1]
        rs_f = best_result[:, 2]

        min_x = np.min(xs_f - rs_f)
        max_x = np.max(xs_f + rs_f)
        min_y = np.min(ys_f - rs_f)
        max_y = np.max(ys_f + rs_f)
        w_eff = max_x - min_x
        h_eff = max_y - min_y

        # Shift to origin
        xs_f = xs_f - min_x
        ys_f = ys_f - min_y

        # Polish
        xs_p, ys_p, rs_p, val_p = _slsqp_polish(xs_f, ys_f, rs_f, w_eff, h_eff, deadline)
        if val_p > best_value:
            best_value = val_p
            best_result = np.column_stack([xs_p, ys_p, rs_p])

    # Fallback
    if best_result is None:
        xs, ys, ww, hh = _make_grid(5, 5)
        r = 0.08
        best_result = np.column_stack([xs, ys, np.full(N, r)])
        best_value = N * r

    # Scale to ensure perimeter <= 4
    xs_f = best_result[:, 0]
    ys_f = best_result[:, 1]
    rs_f = best_result[:, 2]

    perim = _bbox_perim(xs_f, ys_f, rs_f)
    if perim > 4.0 + 1e-9:
        scale = 4.0 / perim
        cx = np.mean(xs_f)
        cy = np.mean(ys_f)
        xs_f = cx + (xs_f - cx) * scale
        ys_f = cy + (ys_f - cy) * scale
        rs_f = rs_f * scale

    return np.column_stack([xs_f, ys_f, rs_f])
