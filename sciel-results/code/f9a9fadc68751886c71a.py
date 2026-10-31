"""Circle packing in perimeter-4 rectangle: multi-start SLSQP with smooth penalties + polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-9):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.5:
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


def _penalty_objective(z, penalty_weight=5000.0):
    """Smooth penalty objective for SLSQP."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.25 or h < 0.25:
        return 1e6

    obj = -np.sum(r)

    # Pairwise overlap penalty (smooth)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    obj += penalty_weight * np.sum(overlap * overlap)

    # Boundary penalty
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)

    obj += penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )

    # Aspect ratio soft constraint
    if w < 0.5:
        obj += penalty_weight * 0.5 * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * 0.5 * (w - 1.5) ** 2

    return obj


def _force_relax(z, max_iter=300, dt_init=0.004):
    """Force-directed relaxation to resolve overlaps."""
    z = z.copy().astype(float)
    n = N
    dt = dt_init

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

        # Wall repulsion
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 50.0
        fx += right_viol * 50.0
        fy -= bottom_viol * 50.0
        fy += top_viol * 50.0

        # Pairwise repulsion
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

        if max_overlap > 1e-9:
            f_pairs = overlap * 10.0
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

        if max_overlap < 1e-8:
            break

        # Adaptive step
        force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
        if force_mag > 0.5:
            dt = max(0.0005, dt * 0.7)
        elif force_mag < 0.05:
            dt = min(0.01, dt * 1.2)

        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        # Clamp
        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], max(r[i], w - r[i]))
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], max(r[i], h - r[i]))

    return z


def _expand_and_relocate(z, time_budget):
    """Iteratively expand each circle to max radius, then relocate to create space."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    max_rounds = 20
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        improved = False
        for i in range(n):
            # Compute max radius for circle i
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
                r[i] = r_max
                improved = True

            # Try to relocate circle i to gain more space
            # Find the most constraining direction
            min_gap = float('inf')
            best_move = (0.0, 0.0)

            # Wall gaps
            gaps = [
                (x[i] - r[i], (1.0, 0.0)),
                (w - x[i] - r[i], (-1.0, 0.0)),
                (y[i] - r[i], (0.0, 1.0)),
                (h - y[i] - r[i], (0.0, -1.0)),
            ]
            for gap, mv in gaps:
                if gap < min_gap:
                    min_gap = gap
                    best_move = mv

            # Neighbor gaps
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                gap = d - (r[i] + r[j])
                if gap < min_gap:
                    min_gap = gap
                    if d > 1e-8:
                        best_move = (dx / d, dy / d)

            # Try moving in the best direction
            if min_gap < 0.03:
                for step in [0.04, 0.02, 0.01, 0.005, 0.002]:
                    nx = x[i] + best_move[0] * step
                    ny = y[i] + best_move[1] * step
                    if _is_valid_move(x, y, r, w, h, i, nx, ny):
                        x[i] = nx
                        y[i] = ny
                        improved = True
                        break
                else:
                    # Try perpendicular
                    px, py = -best_move[1], best_move[0]
                    for step in [0.02, 0.01, 0.005]:
                        for sign in [1.0, -1.0]:
                            nx = x[i] + sign * px * step
                            ny = y[i] + sign * py * step
                            if _is_valid_move(x, y, r, w, h, i, nx, ny):
                                x[i] = nx
                                y[i] = ny
                                improved = True
                                break
                        if improved:
                            break

        if not improved:
            break

    # Final uniform growth
    for _ in range(200):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0005
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


def _is_valid_move(x, y, r, w, h, i, new_x, new_y):
    """Check if moving circle i to new position is valid."""
    if new_x < r[i] - 1e-10 or new_x > w - r[i] + 1e-10:
        return False
    if new_y < r[i] - 1e-10 or new_y > h - r[i] + 1e-10:
        return False
    for j in range(N):
        if i == j:
            continue
        dx = new_x - x[j]
        dy = new_y - y[j]
        d2 = dx * dx + dy * dy
        rs = r[i] + r[j]
        if d2 < rs * rs - 1e-9:
            return False
    return True


def _polish_positions(z, time_budget):
    """Fine-tune positions via force-directed method to maximize radii."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dt = 0.0005
    max_iter = int(min(5000, time_budget * 200))

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 20.0
        fx += right_viol * 20.0
        fy -= bottom_viol * 20.0
        fy += top_viol * 20.0

        # Pairwise repulsion
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) > 1e-10:
            f_pairs = overlap * 8.0
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

        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-10)
        if max_force > 0.05:
            dt = 0.0003
        else:
            dt = 0.0005

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Grow radii
    for _ in range(300):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0003
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


def _constraint_slsqp(z, time_budget):
    """SLSQP with explicit inequality constraints for high-precision refinement."""
    n = N

    def obj(zv):
        r = zv[2::3][:n]
        return -np.sum(r)

    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 1.0), (0.001, 1.0), (0.001, 0.5)])
    bounds.append((0.4, 1.6))

    constraints = []

    # Boundary constraints: x_i - r_i >= 0, w - x_i - r_i >= 0, etc.
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[-1] - p[3*i] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: (2.0 - p[-1]) - p[3*i+1] - p[3*i+2]})

    # Pairwise non-overlap: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (p[3*i] - p[3*j])**2 + (p[3*i+1] - p[3*j+1])**2 - (p[3*i+2] + p[3*j+2])**2
            })

    maxiter = int(min(1000, time_budget * 100))
    try:
        res = minimize(obj, z, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-15, 'maxls': 120})

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


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin

    best_value = 0.0
    best_result = None

    def try_update(z):
        nonlocal best_value, best_result
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w
        if _is_valid(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = z.copy()
            return True
        return False

    # Generate diverse initial guesses
    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))

        # Grid-based starts with various aspect ratios
        aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.7, 1.4, 0.65, 1.6, 0.6, 1.8, 0.55, 2.0, 0.5, 1.3, 0.75, 1.5, 0.67]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.48
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        # Random starts
        rng = np.random.default_rng(123)
        for _ in range(15):
            w0 = rng.uniform(0.55, 1.45)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.06, 0.14)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        return guesses

    guesses = get_initial_guesses()

    # Phase 1: Multi-start SLSQP with penalty
    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 30:
            break

        z0_relaxed = _force_relax(z0, max_iter=200)

        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(400, max(100, remaining * 30)))
        try:
            res = minimize(_penalty_objective, z0_relaxed, method='SLSQP',
                           bounds=[(0.01, 1.5)] * (3 * N) + [(0.4, 1.6)],
                           options={'maxiter': maxiter, 'ftol': 1e-13, 'maxls': 60})

            z_opt = res.x
            try_update(z_opt)
        except Exception:
            continue

    # Phase 2: Polish best result
    if best_result is not None and time.time() < deadline - 20:
        # Position polish
        remaining = deadline - (time.time() - start_time) - 10
        if remaining > 5:
            polished = _polish_positions(best_result, deadline - (time.time() - start_time) - 10)
            if polished is not None:
                try_update(polished)

        # Expansion and relocation
        if time.time() < deadline - 15:
            expanded = _expand_and_relocate(best_result, deadline - (time.time() - start_time) - 8)
            if expanded is not None:
                try_update(expanded)

        # Constraint-based SLSQP refinement
        if time.time() < deadline - 12:
            refined = _constraint_slsqp(best_result, deadline - (time.time() - start_time) - 6)
            if refined is not None:
                try_update(refined)

        # Another round of polish after refinement
        if time.time() < deadline - 5:
            polished2 = _polish_positions(best_result, deadline - (time.time() - start_time) - 2)
            if polished2 is not None:
                try_update(polished2)

        # Final expansion pass
        if time.time() < deadline - 3:
            expanded2 = _expand_and_relocate(best_result, deadline - (time.time() - start_time))
            if expanded2 is not None:
                try_update(expanded2)

    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(N):
            row, col = divmod(i, side)
            cx = (col + 0.5) / side
            cy = (row + 0.5) / side
            r0 = 0.5 / side - 0.01
            cells.append([cx, cy, r0])
        return np.array(cells)

    # Final safety: minimal shrink to ensure strict validity
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    for _ in range(20):
        r_trial = r * (1 - 1e-10)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
