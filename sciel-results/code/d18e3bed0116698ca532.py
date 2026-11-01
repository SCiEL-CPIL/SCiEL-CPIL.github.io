"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii.
Grid-based init + force-directed relaxation + multi-stage SLSQP polish.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
TOL = 1e-9


def _objective(z, penalty_weight=1000.0):
    """Vectorized objective with smooth penalties for overlap and wall violations."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.25 or h < 0.25:
        return 1e8
    if np.any(r <= 0):
        return 1e8

    obj = -np.sum(r)

    # Pairwise overlap penalty (smooth)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-15))
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    obj += penalty_weight * np.sum(overlap * overlap)

    # Wall constraint penalties (vectorized)
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)

    obj += penalty_weight * (np.sum(left_viol ** 2) + np.sum(right_viol ** 2) +
                             np.sum(bottom_viol ** 2) + np.sum(top_viol ** 2))

    # Penalize tiny radii
    small_r = np.maximum(0.01 - r, 0.0)
    obj += penalty_weight * np.sum(small_r ** 2)

    # Keep aspect ratio reasonable
    if w < 0.4:
        obj += penalty_weight * (0.4 - w) ** 2
    elif w > 1.6:
        obj += penalty_weight * (w - 1.6) ** 2

    return obj


def _check_validity(x, y, r, w, h, tol=TOL):
    """Check packing validity. Returns (valid, sum_of_radii)."""
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
            if d2 < r_sum * r_sum - tol:
                return False, 0.0

    return True, float(np.sum(r))


def _force_relax(x, y, r, w, h, max_iter=800, dt=0.003):
    """
    Force-directed relaxation: push overlapping circles apart, pull toward
    wall-free region. Returns updated (x, y, r, w, h).
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)

    for it in range(max_iter):
        fx = np.zeros(N)
        fy = np.zeros(N)

        # Pairwise repulsion for overlapping circles
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                if d < min_d:
                    if d < 1e-12:
                        # Nearly coincident: push in random direction
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle)
                        dy = np.sin(angle)
                        d = 1e-6
                    force = (min_d - d) / d
                    fx[i] += force * dx
                    fy[i] += force * dy
                    fx[j] -= force * dx
                    fy[j] -= force * dy

        # Wall repulsion
        for i in range(N):
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * 5.0
            elif x[i] > w - r[i]:
                fx[i] -= (x[i] - (w - r[i])) * 5.0
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * 5.0
            elif y[i] > h - r[i]:
                fy[i] -= (y[i] - (h - r[i])) * 5.0

        # Apply forces with adaptive step
        for i in range(N):
            x[i] += dt * fx[i]
            y[i] += dt * fy[i]

            # Hard clamp to walls
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])

        # Decay step size
        dt *= 0.999

        # Convergence check: if no overlaps remain, stop
        has_overlap = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                if dx * dx + dy * dy < (r[i] + r[j]) ** 2:
                    has_overlap = True
                    break
            if has_overlap:
                break
        if not has_overlap:
            break

    return x, y, r, w, h


def _grid_init(rows, cols, w, h):
    """
    Grid-based cellular decomposition: partition rectangle into rows x cols cells,
    assign circles to cells with radii based on cell inradius.
    Returns (x, y, r) arrays.
    """
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)

    cell_w = w / cols
    cell_h = h / rows

    idx = 0
    for ri in range(rows):
        for ci in range(cols):
            if idx >= N:
                break
            cx = (ci + 0.5) * cell_w
            cy = (ri + 0.5) * cell_h
            max_r = min(cell_w, cell_h) / 2.0 * 0.85

            remaining_circles = N - idx
            remaining_cells = (rows - ri) * cols - ci

            if remaining_circles > remaining_cells:
                # Need 2 circles in this cell
                r1 = max_r * 0.55
                r2 = max_r * 0.35
                if cell_w >= cell_h:
                    offset = (r1 + r2) * 0.55
                    x[idx] = cx - offset
                    y[idx] = cy
                    r[idx] = max(r1, 0.01)
                    idx += 1
                    if idx < N:
                        x[idx] = cx + offset
                        y[idx] = cy
                        r[idx] = max(r2, 0.01)
                        idx += 1
                else:
                    offset = (r1 + r2) * 0.55
                    x[idx] = cx
                    y[idx] = cy - offset
                    r[idx] = max(r1, 0.01)
                    idx += 1
                    if idx < N:
                        x[idx] = cx
                        y[idx] = cy + offset
                        r[idx] = max(r2, 0.01)
                        idx += 1
            else:
                x[idx] = cx
                y[idx] = cy
                r[idx] = max(max_r, 0.01)
                idx += 1

    # Fill remaining with tiny circles
    while idx < N:
        x[idx] = w / 2.0
        y[idx] = h / 2.0
        r[idx] = 0.01
        idx += 1

    return x, y, r


def _get_initial_guesses():
    """Generate diverse valid initial guesses using grid decomposition."""
    guesses = []

    grid_configs = [
        (3, 7), (7, 3),
        (4, 6), (6, 4),
        (5, 5),
        (3, 8), (8, 3),
        (4, 5), (5, 4),
        (2, 11), (11, 2),
        (3, 5), (5, 3),
        (6, 5), (5, 6),
    ]

    aspect_ratios = [
        1.0, 1.2, 0.8, 1.5, 0.5, 1.3, 0.7, 1.4, 0.6, 1.1, 0.9,
        1.6, 0.4, 1.7, 0.35, 1.8, 1.25, 0.75, 1.35, 0.65
    ]

    for w0 in aspect_ratios:
        h0 = 2.0 - w0
        if w0 < 0.25 or h0 < 0.25:
            continue
        for rows, cols in grid_configs:
            x, y, r = _grid_init(rows, cols, w0, h0)
            valid, _ = _check_validity(x, y, r, w0, h0)
            if valid:
                guesses.append((w0, h0, x.copy(), y.copy(), r.copy()))

    # Sort by initial sum of radii (descending)
    guesses.sort(key=lambda g: -np.sum(g[4]))

    # Deduplicate: keep top diverse set
    seen = set()
    unique = []
    for g in guesses:
        key = (round(g[0], 2), round(g[1], 2), g[2].tobytes()[:100])
        if key not in seen:
            seen.add(key)
            unique.append(g)
        if len(unique) >= 40:
            break

    return unique


def _slsqp_polish(x, y, r, w, h, max_iter=500, ftol=1e-12, penalty=1000.0):
    """SLSQP optimization over (x, y, r, w) with smooth penalty objective."""
    z0 = np.concatenate([x, y, r, [w]])

    bounds = []
    for i in range(N):
        bounds.append((0.001, max(0.002, w - 0.001)))
        bounds.append((0.001, max(0.002, h - 0.001)))
        bounds.append((0.005, min(w, h) / 2.0 - 0.001))
    bounds.append((0.25, 1.75))

    try:
        res = minimize(
            lambda z: _objective(z, penalty),
            z0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'maxls': 100}
        )
        z_opt = res.x
        x_opt = z_opt[0::3][:N]
        y_opt = z_opt[1::3][:N]
        r_opt = z_opt[2::3][:N]
        w_opt = z_opt[-1]
        h_opt = 2.0 - w_opt

        valid, val = _check_validity(x_opt, y_opt, r_opt, w_opt, h_opt)
        if valid and val > 0:
            return x_opt, y_opt, r_opt, w_opt, h_opt, val
    except Exception:
        pass

    return None


def _tight_polish(x, y, r, w, h, max_iter=800, ftol=1e-14, penalty=3000.0):
    """Tight-constraint SLSQP polish with narrow bounds around current solution."""
    z0 = np.concatenate([x, y, r, [w]])

    bounds = []
    for i in range(N):
        bounds.append((max(0.001, x[i] - 0.05), min(w - 0.001, x[i] + 0.05)))
        bounds.append((max(0.001, y[i] - 0.05), min(h - 0.001, y[i] + 0.05)))
        bounds.append((max(0.005, r[i] * 0.85), min(min(w, h) / 2.0 - 0.001, r[i] * 1.15)))
    bounds.append((max(0.25, w - 0.1), min(1.75, w + 0.1)))

    try:
        res = minimize(
            lambda z: _objective(z, penalty),
            z0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'maxls': 80}
        )
        z_opt = res.x
        x_opt = z_opt[0::3][:N]
        y_opt = z_opt[1::3][:N]
        r_opt = z_opt[2::3][:N]
        w_opt = z_opt[-1]
        h_opt = 2.0 - w_opt

        valid, val = _check_validity(x_opt, y_opt, r_opt, w_opt, h_opt)
        if valid and val > 0:
            return x_opt, y_opt, r_opt, w_opt, h_opt, val
    except Exception:
        pass

    return None


def _expansion_polish(x, y, r, w, h, deadline):
    """
    Iteratively try to expand each circle's radius slightly, adjusting position
    if needed to maintain validity.
    """
    x = x.copy()
    y = y.copy()
    r = r.copy()

    for iteration in range(60):
        if time.time() > deadline:
            break

        improved = False
        for i in range(N):
            old_r = r[i]
            for dr in [0.004, 0.002, 0.001, 0.0005]:
                new_r = old_r + dr
                if new_r > min(w, h) / 2.0 - 0.001:
                    continue

                found = False
                for dx in [0, 0.004, -0.004, 0.008, -0.008, 0.015, -0.015]:
                    for dy in [0, 0.004, -0.004, 0.008, -0.008, 0.015, -0.015]:
                        nx = x[i] + dx
                        ny = y[i] + dy

                        if nx < new_r or nx > w - new_r:
                            continue
                        if ny < new_r or ny > h - new_r:
                            continue

                        valid = True
                        for j in range(N):
                            if i == j:
                                continue
                            ddx = nx - x[j]
                            ddy = ny - y[j]
                            if ddx * ddx + ddy * ddy < (new_r + r[j]) ** 2 - TOL:
                                valid = False
                                break

                        if valid:
                            x[i] = nx
                            y[i] = ny
                            r[i] = new_r
                            found = True
                            break
                    if found:
                        break

                if found:
                    improved = True
                    break

        if not improved:
            break

    valid, val = _check_validity(x, y, r, w, h)
    if valid:
        return x, y, r, w, h, val
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None

    def update_best(x, y, r, w, h, val):
        nonlocal best_value, best_result
        if val > best_value:
            best_value = val
            best_result = np.column_stack([x, y, r])

    # Phase 1: Generate initial guesses and run force-relax + SLSQP
    guesses = _get_initial_guesses()

    for idx, (w0, h0, x0, y0, r0) in enumerate(guesses):
        if time.time() > deadline - 25.0:
            break

        # Force-directed relaxation to resolve any residual overlaps
        x_f, y_f, r_f, w_f, h_f = _force_relax(x0, y0, r0, w0, h0, max_iter=400)

        # SLSQP polish
        remaining = deadline - (time.time() - start_time) - 20.0
        maxiter = int(min(500, max(100, remaining * 20)))
        result = _slsqp_polish(x_f, y_f, r_f, w_f, h_f, max_iter=maxiter, penalty=1000.0)

        if result is not None:
            xr, yr, rr, wr, hr, val = result
            update_best(xr, yr, rr, wr, hr, val)

            # Tight polish on the best
            if time.time() < deadline - 15.0:
                tight = _tight_polish(xr, yr, rr, wr, hr, max_iter=400, ftol=1e-14, penalty=3000.0)
                if tight is not None:
                    xt, yt, rt, wt, ht, vt = tight
                    update_best(xt, yt, rt, wt, ht, vt)

        # Also try expansion on the force-relaxed result
        if time.time() < deadline - 10.0:
            exp = _expansion_polish(x_f, y_f, r_f, w_f, h_f, deadline - 5.0)
            if exp is not None:
                xe, ye, re, we, he, ve = exp
                update_best(xe, ye, re, we, he, ve)

    # Phase 2: Final polish on the global best
    if best_result is not None and time.time() < deadline - 10.0:
        x_best = best_result[:, 0]
        y_best = best_result[:, 1]
        r_best = best_result[:, 2]

        # Recompute w, h from bounding box (or use the stored aspect)
        # We need to recover w and h; store them alongside
        # For now, re-derive: the optimizer varied w, so we need to track it.
        # Let's re-run a tight polish using the best positions.
        # Estimate w, h from the positions and radii
        x_max = np.max(x_best + r_best)
        x_min = np.min(x_best - r_best)
        y_max = np.max(y_best + r_best)
        y_min = np.min(y_best - r_best)

        # Use the w that was used (we need to track this; for safety, try a few)
        # Actually, let's just do a final expansion + force pass
        if time.time() < deadline - 5.0:
            # Try force relax on best
            x_fp, y_fp, r_fp, w_est, h_est = _force_relax(
                x_best, y_best, r_best,
                max(x_max, 1.0), 2.0 - max(x_max, 1.0) if max(x_max, 1.0) < 1.75 else 0.25,
                max_iter=200
            )
            valid, val = _check_validity(x_fp, y_fp, r_fp, max(x_max, 1.0), 2.0 - max(x_max, 1.0) if max(x_max, 1.0) < 1.75 else 0.25)
            if valid:
                update_best(x_fp, y_fp, r_fp, max(x_max, 1.0), 2.0 - max(x_max, 1.0) if max(x_max, 1.0) < 1.75 else 0.25, val)

    # Fallback
    if best_result is None:
        w, h = 1.0, 1.0
        x, y, r = _grid_init(5, 5, w, h)
        best_result = np.column_stack([x, y, r])

    return best_result
