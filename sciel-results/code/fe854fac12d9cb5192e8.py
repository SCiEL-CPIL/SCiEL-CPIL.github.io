"""Circle packing in perimeter-4 rectangle: SLSQP + smooth-gap polish + radial growth."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(r <= 0):
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _is_valid_move(x, y, r, w, h, i, new_x, new_y):
    """Check if moving circle i to new_x, new_y is valid."""
    if new_x < r[i] or new_x > w - r[i]:
        return False
    if new_y < r[i] or new_y > h - r[i]:
        return False
    for j in range(N):
        if i == j:
            continue
        dx = new_x - x[j]
        dy = new_y - y[j]
        d2 = dx * dx + dy * dy
        r_sum = r[i] + r[j]
        if d2 < r_sum * r_sum - 1e-9:
            return False
    return True


def _force_relax_unified(z, max_iter=500):
    """Force-directed relaxation in unified parameter space with adaptive step."""
    z = z.copy().astype(float)
    n = N
    dt = 0.005
    min_dt = 0.0001
    convergence_tol = 1e-8

    for it in range(max_iter):
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w

        if w < 0.1 or h < 0.1:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 100
        fx += right_viol * 100
        fy -= bottom_viol * 100
        fy += top_viol * 100

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        max_overlap = float(np.max(overlap)) if overlap.size > 0 else 0.0

        if max_overlap > convergence_tol:
            f_pairs = overlap * 15.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            for idx in range(len(iu[0])):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                fx[i_idx] += ux[idx] * f_pairs[idx]
                fy[i_idx] += uy[idx] * f_pairs[idx]
                fx[j_idx] -= ux[idx] * f_pairs[idx]
                fy[j_idx] -= uy[idx] * f_pairs[idx]

        if max_overlap < convergence_tol:
            break

        force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
        if force_mag > 1.0:
            dt = max(min_dt, dt * 0.5)
        elif force_mag < 0.1:
            dt = min(0.01, dt * 1.1)

        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

    return z


def _vectorized_objective(z, penalty_weight=1000.0):
    """Vectorized objective with smooth penalties for speed."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    overlap_raw = r_sum_pairs - d_pairs
    overlap = np.maximum(overlap_raw, 0.0)
    overlap_penalty = penalty_weight * np.sum(overlap * overlap)
    obj += overlap_penalty

    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)

    boundary_penalty = penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )
    obj += boundary_penalty

    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def _hexagonal_initial(w, h, n=N):
    """Dense hexagonal grid of circles inside a w x h rectangle. Returns (x, y, r)."""
    # Choose a base radius and lay out a hex lattice, then trim to fit.
    # Target ~n circles. Hex packing density ~ pi/(2*sqrt(3)).
    area = w * h
    # Estimate radius so that n circles of that radius roughly fill area.
    # Area per circle in hex packing = 2*sqrt(3)*r^2
    r_est = np.sqrt(area / (n * 2 * np.sqrt(3)))
    r_est = min(r_est, min(w, h) / 3.0)
    r_est = max(r_est, 0.02)

    x = []
    y = []
    dy = r_est * np.sqrt(3)
    dx = 2 * r_est
    row = 0
    while True:
        y0 = r_est + row * dy
        if y0 + r_est > h:
            break
        offset = (row % 2) * dx / 2
        col = 0
        while True:
            x0 = r_est + offset + col * dx
            if x0 + r_est > w:
                break
            x.append(x0)
            y.append(y0)
            col += 1
        if len(x) >= n + 20:
            break
        row += 1
        if row > 200:
            break

    x = np.array(x)
    y = np.array(y)
    r = np.full_like(x, r_est)

    if len(x) < n:
        # pad with tiny circles in a grid
        side = int(np.ceil(np.sqrt(n)))
        for k in range(n - len(x)):
            row, col = divmod(k, side)
            x = np.append(x, (col + 0.5) / side * w)
            y = np.append(y, (row + 0.5) / side * h)
            r = np.append(r, min(w, h) / side * 0.3)

    # Keep the n most central (or first n). Use first n for determinism.
    x = x[:n]
    y = y[:n]
    r = r[:n]
    return x, y, r


def _radial_growth(x, y, r, w, h, time_budget):
    """Iteratively expand each circle to its locally admissible radius, then relax.

    Deterministic pressure-based expansion: repeatedly, for each circle compute the
    max radius allowed by walls and neighbors, grow toward it, then run a light
    force relaxation to re-space. Returns (x, y, r) if valid.
    """
    n = N
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)

    max_rounds = 40
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break
        # Light relaxation to clear overlaps before measuring clearance.
        for _ in range(40):
            fx = np.zeros(n)
            fy = np.zeros(n)
            left_viol = np.maximum(r - x, 0.0)
            right_viol = np.maximum(x + r - w, 0.0)
            bottom_viol = np.maximum(r - y, 0.0)
            top_viol = np.maximum(y + r - h, 0.0)
            fx -= left_viol * 40
            fx += right_viol * 40
            fy -= bottom_viol * 40
            fy += top_viol * 40
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d_sq = dx * dx + dy * dy
            r_sum = r[:, None] + r[None, :]
            iu = np.triu_indices(n, k=1)
            d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
            overlap = np.maximum(r_sum[iu] - d_pairs, 0.0)
            if np.max(overlap) > 1e-10:
                f_pairs = overlap * 12.0
                dx_pairs = dx[iu]
                dy_pairs = dy[iu]
                ux = dx_pairs / d_pairs
                uy = dy_pairs / d_pairs
                for idx in range(len(iu[0])):
                    ii = iu[0][idx]
                    jj = iu[1][idx]
                    fx[ii] += ux[idx] * f_pairs[idx]
                    fy[ii] += uy[idx] * f_pairs[idx]
                    fx[jj] -= ux[idx] * f_pairs[idx]
                    fy[jj] -= uy[idx] * f_pairs[idx]
            x += 0.001 * fx
            y += 0.001 * fy
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)

        # Grow each circle toward its local clearance.
        grew = False
        for i in range(n):
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)
            target = r[i] + 0.5 * (r_max - r[i])
            if target > r[i] + 1e-7:
                r[i] = target
                grew = True
        if not grew:
            break

    if _is_valid(x, y, r, w, h):
        return x, y, r
    return None


def _polish_result(z, time_budget):
    """Polish with smooth gap-aware repulsion + radius growth."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dt = 0.001
    max_iter = int(min(5000, time_budget * 200))
    gap_smooth_len = 0.05

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 50
        fx += right_viol * 50
        fy -= bottom_viol * 50
        fy += top_viol * 50

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))

        gap = d_pairs - r_sum_pairs
        overlap_force = np.maximum(-gap, 0.0) * 25.0
        soft_force = 0.8 * np.exp(-gap / gap_smooth_len)
        f_pairs = overlap_force + soft_force

        active = f_pairs > 0.01
        if np.any(active):
            idx_active = np.where(active)[0]
            for idx in idx_active:
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                d = d_pairs[idx]
                if d > 1e-6:
                    ux = dx[i_idx, j_idx] / d
                    uy = dy[i_idx, j_idx] / d
                    f = f_pairs[idx]
                    fx[i_idx] += ux * f
                    fy[i_idx] += uy * f
                    fx[j_idx] -= ux * f
                    fy[j_idx] -= uy * f

        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 0.1:
            dt = 0.0004
        elif max_force > 0.01:
            dt = 0.0008
        else:
            dt = 0.0012

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Grow radii uniformly.
    grow_factor = 1.0003
    for _ in range(300):
        if time.time() > time_budget:
            break
        r_trial = r * grow_factor
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _expansion_relaxation_polish(z, time_budget):
    """Iteratively expand circles and shift positions to create space."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    max_rounds = 30
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)

            if r_max > r[i] + 1e-6:
                lo = r[i]
                hi = r_max
                for _ in range(20):
                    mid = (lo + hi) / 2.0
                    if _is_valid_move(x, y, np.where(np.arange(n) == i, mid, r), w, h, i, x[i], y[i]):
                        lo = mid
                    else:
                        hi = mid
                new_r = lo
                if new_r > r[i] + 1e-7:
                    r[i] = new_r
                    improved = True

            min_dist = float('inf')
            blocker = None
            d_left = x[i] - r[i]
            d_right = w - x[i] - r[i]
            d_bottom = y[i] - r[i]
            d_top = h - y[i] - r[i]
            if d_left < min_dist:
                min_dist = d_left
                blocker = 'left'
            if d_right < min_dist:
                min_dist = d_right
                blocker = 'right'
            if d_bottom < min_dist:
                min_dist = d_bottom
                blocker = 'bottom'
            if d_top < min_dist:
                min_dist = d_top
                blocker = 'top'
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                gap = d - (r[i] + r[j])
                if gap < min_dist:
                    min_dist = gap
                    blocker = j

            if min_dist < 0.02:
                move_x = 0.0
                move_y = 0.0
                if blocker == 'left':
                    move_x = 1.0
                elif blocker == 'right':
                    move_x = -1.0
                elif blocker == 'bottom':
                    move_y = 1.0
                elif blocker == 'top':
                    move_y = -1.0
                else:
                    j = blocker
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d > 1e-6:
                        move_x = dx / d
                        move_y = dy / d

                moved = False
                for step in [0.05, 0.03, 0.02, 0.01, 0.005, 0.002]:
                    new_x = x[i] + move_x * step
                    new_y = y[i] + move_y * step
                    if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                        x[i] = new_x
                        y[i] = new_y
                        improved = True
                        moved = True
                        break
                if not moved:
                    perp_x = -move_y
                    perp_y = move_x
                    for step in [0.03, 0.01, 0.005]:
                        for sign in [1.0, -1.0]:
                            new_x = x[i] + sign * perp_x * step
                            new_y = y[i] + sign * perp_y * step
                            if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                                x[i] = new_x
                                y[i] = new_y
                                improved = True
                                moved = True
                                break
                        if moved:
                            break

        if not improved:
            break

    grow_factor = 1.001
    for _ in range(100):
        if time.time() > time_budget:
            break
        r_trial = r * grow_factor
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _constraint_refine(z, time_budget):
    """High-precision SLSQP refinement using explicit constraints."""
    z = z.copy().astype(float)
    n = N

    def obj_refine(zv):
        r = zv[2::3][:n]
        return -np.sum(r)

    constraints = []
    bounds = []
    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3 * i] - p[3 * i + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[-1] - p[3 * i] - p[3 * i + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3 * i + 1] - p[3 * i + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: (2.0 - p[-1]) - p[3 * i + 1] - p[3 * i + 2]})
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (
                    (p[3 * i] - p[3 * j]) ** 2 +
                    (p[3 * i + 1] - p[3 * j + 1]) ** 2 -
                    (p[3 * i + 2] + p[3 * j + 2]) ** 2
                )
            })

    maxiter = int(min(800, time_budget * 150))
    try:
        res = minimize(obj_refine, z, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})
        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w
        if _is_valid(x, y, r, w, h):
            return z_opt
        return None
    except Exception:
        return None


def _epigraph_refine(z, time_budget):
    """Smooth log-sum-exp epigraph refinement: minimize -sum(r) + soft-violation.

    Uses a smooth surrogate for the nonsmooth overlap/wall constraints so the
    optimizer can escape local minima that the hard non-convex distance
    constraints trap. Returns a valid z if it improves.
    """
    z = z.copy().astype(float)
    n = N
    alpha = 40.0  # inverse temperature for soft-max (larger = sharper)

    def soft_violation(zv):
        x = zv[0::3][:n]
        y = zv[1::3][:n]
        r = zv[2::3][:n]
        w = zv[-1]
        h = 2.0 - w
        # Pairwise: log-sum-exp of (r_i+r_j - d)^2 over pairs (soft max of violations)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
        viol = np.maximum(r_sum[iu] - d_pairs, 0.0)
        # Smooth max via log-sum-exp (in squared space)
        v2 = viol * viol
        v2 = np.clip(v2, 1e-12, None)
        m = np.max(v2)
        soft_max = m + np.log(np.sum(np.exp(v2 - m)))
        # Wall violations
        lv = np.maximum(r - x, 0.0)
        rv = np.maximum(x + r - w, 0.0)
        bv = np.maximum(r - y, 0.0)
        tv = np.maximum(y + r - h, 0.0)
        wall2 = np.concatenate([lv * lv, rv * rv, bv * bv, tv * tv])
        wall2 = np.clip(wall2, 1e-12, None)
        mw = np.max(wall2)
        soft_wall = mw + np.log(np.sum(np.exp(wall2 - mw)))
        return soft_max + soft_wall

    def obj_epi(zv):
        r = zv[2::3][:n]
        return -np.sum(r) + alpha * soft_violation(zv)

    bounds = []
    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    maxiter = int(min(500, time_budget * 80))
    try:
        res = minimize(obj_epi, z, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 80})
        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w
        if _is_valid(x, y, r, w, h):
            return z_opt
        return None
    except Exception:
        return None


def _sequential_placement(w, h, time_budget):
    """Sequential placement: place circles one by one, each maximizing its radius."""
    n = N
    x = np.zeros(n)
    y = np.zeros(n)
    r = np.zeros(n)
    side = int(np.ceil(np.sqrt(n)))
    for i in range(n):
        row, col = divmod(i, side)
        x[i] = (col + 0.5) / side * w
        y[i] = (row + 0.5) / side * h
        r[i] = min(w / side, h / side) * 0.4

    orderings = [
        list(range(n)),
        sorted(range(n), key=lambda i: -min(x[i], w - x[i], y[i], h - y[i])),
        sorted(range(n), key=lambda i: min(x[i], w - x[i], y[i], h - y[i])),
    ]

    best_x = x.copy()
    best_y = y.copy()
    best_r = r.copy()
    best_val = 0.0

    for order in orderings:
        if time.time() > time_budget:
            break
        x_trial = np.zeros(n)
        y_trial = np.zeros(n)
        r_trial = np.zeros(n)
        placed_trial = np.zeros(n, dtype=bool)

        for idx, ci in enumerate(order):
            cx = x[ci]
            cy = y[ci]
            cr = r[ci]

            r_wall = min(cx, w - cx, cy, h - cy)
            r_max = r_wall
            for j in range(n):
                if placed_trial[j]:
                    dx = cx - x_trial[j]
                    dy = cy - y_trial[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    r_allowed = d - r_trial[j]
                    if r_allowed < r_max:
                        r_max = r_allowed
            r_max = max(0.0, r_max)

            best_local_r = 0.0
            best_local_x = cx
            best_local_y = cy

            step = 0.05
            for dx in np.arange(-0.2, 0.21, step):
                for dy in np.arange(-0.2, 0.21, step):
                    nx = cx + dx
                    ny = cy + dy
                    if nx < 0.01 or nx > w - 0.01 or ny < 0.01 or ny > h - 0.01:
                        continue
                    rw = min(nx, w - nx, ny, h - ny)
                    rm = rw
                    for j in range(n):
                        if placed_trial[j]:
                            ddx = nx - x_trial[j]
                            ddy = ny - y_trial[j]
                            d = np.sqrt(ddx * ddx + ddy * ddy)
                            r_allowed = d - r_trial[j]
                            if r_allowed < rm:
                                rm = r_allowed
                    rm = max(0.0, rm)
                    if rm > best_local_r:
                        best_local_r = rm
                        best_local_x = nx
                        best_local_y = ny

            rw = min(cx, w - cx, cy, h - cy)
            rm = rw
            for j in range(n):
                if placed_trial[j]:
                    ddx = cx - x_trial[j]
                    ddy = cy - y_trial[j]
                    d = np.sqrt(ddx * ddx + ddy * ddy)
                    r_allowed = d - r_trial[j]
                    if r_allowed < rm:
                        rm = r_allowed
            rm = max(0.0, rm)
            if rm > best_local_r:
                best_local_r = rm
                best_local_x = cx
                best_local_y = cy

            x_trial[ci] = best_local_x
            y_trial[ci] = best_local_y
            r_trial[ci] = best_local_r
            placed_trial[ci] = True

        val = float(np.sum(r_trial))
        if val > best_val:
            best_val = val
            best_x = x_trial.copy()
            best_y = y_trial.copy()
            best_r = r_trial.copy()

    if best_val <= 0:
        return None

    z_new = np.empty(3 * N + 1)
    z_new[0::3][:n] = best_x
    z_new[1::3][:n] = best_y
    z_new[2::3][:n] = best_r
    z_new[-1] = w
    return z_new


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_value = 0.0
    best_result = None

    penalty_weight = 1000.0

    def objective(z):
        return _vectorized_objective(z, penalty_weight)

    def check_validity(z, tol=1e-9):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w
        if w <= 0.05 or h <= 0.05:
            return False, 0.0
        for i in range(N):
            if r[i] <= 0:
                return False, 0.0
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False, 0.0
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False, 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum ** 2 - tol:
                    return False, 0.0
        val = float(np.sum(r))
        return True, val

    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))

        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.45
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        # Hexagonal radial-growth initializers (deterministic dense start).
        for w0 in [1.0, 0.9, 1.1, 0.8, 1.2]:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            hx, hy, hr = _hexagonal_initial(w0, h0, N)
            z = np.zeros(3 * N + 1)
            z[0::3][:N] = hx
            z[1::3][:N] = hy
            z[2::3][:N] = hr
            z[-1] = w0
            guesses.append(z)

        rng = np.random.default_rng(42)
        for _ in range(10):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.05, 0.12)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        return guesses

    guesses = get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 20:
            break

        z0_relaxed = _force_relax_unified(z0, max_iter=500)

        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(600, max(150, remaining * 50)))
        try:
            res = minimize(objective, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})
            z_opt = res.x
            valid, val = check_validity(z_opt)
            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
            elif not valid:
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.95
                valid, val = check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair.copy()
        except Exception:
            continue

    # --- Phase 2: Polish on best result ---
    if best_result is not None and time.time() < deadline - 25:
        polished = _polish_result(best_result, deadline - (time.time() - start_time) - 15)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        if best_result is not None and time.time() < deadline - 15:
            expanded = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 10)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

            if best_result is not None and time.time() < deadline - 10:
                expanded2 = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 5)
                if expanded2 is not None:
                    valid, val = check_validity(expanded2)
                    if valid and val > best_value:
                        best_value = val
                        best_result = expanded2

        # Smooth epigraph refine (escape local minima), then a hard constraint refine.
        if best_result is not None and time.time() < deadline - 14:
            epi = _epigraph_refine(best_result, deadline - (time.time() - start_time) - 8)
            if epi is not None:
                valid, val = check_validity(epi)
                if valid and val > best_value:
                    best_value = val
                    best_result = epi

        if best_result is not None and time.time() < deadline - 12:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 6)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        if best_result is not None and time.time() < deadline - 4:
            polished2 = _polish_result(best_result, deadline - (time.time() - start_time))
            if polished2 is not None:
                valid, val = check_validity(polished2)
                if valid and val > best_value:
                    best_value = val
                    best_result = polished2

    # --- Phase 3: Sequential placement as alternative approach ---
    if time.time() < deadline - 10:
        for w0 in [1.0, 0.9, 1.1, 0.8, 1.2]:
            if time.time() > deadline - 5:
                break
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            seq_result = _sequential_placement(w0, h0, deadline - (time.time() - start_time) - 3)
            if seq_result is not None:
                valid, val = check_validity(seq_result)
                if valid and val > best_value:
                    best_value = val
                    best_result = seq_result

    # --- Phase 4: Radial growth from hexagonal grid as a deterministic alternative ---
    if time.time() < deadline - 8:
        for w0 in [1.0, 0.9, 1.1, 0.8, 1.2]:
            if time.time() > deadline - 4:
                break
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            hx, hy, hr = _hexagonal_initial(w0, h0, N)
            grown = _radial_growth(hx, hy, hr, w0, h0, deadline - (time.time() - start_time) - 2)
            if grown is not None:
                gx, gy, gr = grown
                z_new = np.empty(3 * N + 1)
                z_new[0::3][:N] = gx
                z_new[1::3][:N] = gy
                z_new[2::3][:N] = gr
                z_new[-1] = w0
                valid, val = check_validity(z_new)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_new

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
