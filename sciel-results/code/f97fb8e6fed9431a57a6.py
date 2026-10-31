"""Circle packing in perimeter-4 rectangle: unified SLSQP with free aspect ratio."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    best_value = 0.0
    best_result = None

    # --- Phase 1: Unified SLSQP optimization with free aspect ratio ---
    # Parameter vector: [x1, y1, r1, ..., xN, yN, rN, w]
    # h = 2.0 - w

    n_starts = 12
    penalty_weight = 500.0

    def _vectorized_objective(z):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w < 0.3 or h < 0.3:
            return 1e6

        obj = -np.sum(r)

        # Vectorized overlap penalty
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx * dx + dy * dy)
        dist_req = r[:, None] + r[None, :]
        # Only upper triangle
        iu = np.triu_indices(N, 1)
        overlaps = np.maximum(0, dist_req[iu] - d[iu])
        obj += penalty_weight * np.sum(overlaps ** 2)

        # Vectorized boundary penalty
        bx = np.maximum(0, r - x) + np.maximum(0, x - (w - r))
        by = np.maximum(0, r - y) + np.maximum(0, y - (h - r))
        obj += penalty_weight * (np.sum(bx ** 2) + np.sum(by ** 2))

        # Keep aspect ratio in reasonable range
        if w < 0.5:
            obj += penalty_weight * (0.5 - w) ** 2
        elif w > 1.5:
            obj += penalty_weight * (w - 1.5) ** 2

        return obj

    def objective(z):
        return _vectorized_objective(z)

    def check_validity(z, tol=1e-8):
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

    # Generate diverse initial guesses
    def get_initial_guesses():
        guesses = []

        # Grid-based starts with varying aspect ratios
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5]
        side = int(np.ceil(np.sqrt(N)))

        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = []
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.45
                z.extend([cx, cy, r0])
            z.append(w0)
            guesses.append(np.array(z))

        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(n_starts - len(guesses)):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = []
            for i in range(N):
                cx = rng.uniform(0.1, w0 - 0.1)
                cy = rng.uniform(0.1, h0 - 0.1)
                r0 = rng.uniform(0.05, 0.12)
                z.extend([cx, cy, r0])
            z.append(w0)
            guesses.append(np.array(z))

        return guesses

    guesses = get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 5:
            break

        # Force-directed relaxation to resolve initial overlaps
        z0_relaxed = _force_relax_unified(z0, max_iter=200)

        # SLSQP optimization
        maxiter = int(min(300, (deadline - (time.time() - start_time)) * 30))
        try:
            res = minimize(objective, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-10, 'maxls': 50})

            z_opt = res.x
            valid, val = check_validity(z_opt)

            if valid and val > best_value:
                best_value = val
                best_result = z_opt
            elif not valid:
                # Try repair: shrink radii slightly
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.95
                valid, val = check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair

        except Exception:
            continue

    # --- Phase 2: Constraint-based refinement ---
    if best_result is not None and time.time() < deadline - 15:
        refined = _constraint_refine(best_result, deadline - (time.time() - start_time))
        if refined is not None:
            valid, val = check_validity(refined)
            if valid and val > best_value:
                best_value = val
                best_result = refined

    # --- Phase 3: Local polish on best result ---
    if best_result is not None and time.time() < deadline - 8:
        polished = _polish_result(best_result, deadline - (time.time() - start_time))
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

    # --- Phase 4: High-precision SLSQP refinement ---
    if best_result is not None and time.time() < deadline - 3:
        refined = _high_precision_refine(best_result, deadline - (time.time() - start_time))
        if refined is not None:
            valid, val = check_validity(refined)
            if valid and val > best_value:
                best_value = val
                best_result = refined

    if best_result is None:
        # Fallback: simple grid in unit square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Format output
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]

    # Safety shrink
    r = r * (1 - 1e-6)

    return np.column_stack((x, y, r))


def _force_relax_unified(z, max_iter=200):
    """Force-directed relaxation in unified parameter space with adaptive step size."""
    z = z.copy().astype(float)
    n = N
    dt = 0.003
    dt_min = 0.0005
    dt_decay = 0.995

    for _ in range(max_iter):
        x = z[0::3][:n].copy()
        y = z[1::3][:n].copy()
        r = z[2::3][:n].copy()
        w = z[-1]
        h = 2.0 - w

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        for i in range(n):
            if x[i] - r[i] < 0:
                fx[i] += (x[i] - r[i]) * 20
            if x[i] + r[i] > w:
                fx[i] -= (x[i] + r[i] - w) * 20
            if y[i] - r[i] < 0:
                fy[i] += (y[i] - r[i]) * 20
            if y[i] + r[i] > h:
                fy[i] -= (y[i] + r[i] - h) * 20

        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-12:
                    overlap = min_d - d
                    fx[i] += (dx / d) * overlap * 10
                    fy[i] += (dy / d) * overlap * 10
                    fx[j] -= (dx / d) * overlap * 10
                    fy[j] -= (dy / d) * overlap * 10

        # Adaptive step size: reduce if forces are large
        max_force = max(np.max(np.abs(fx)), np.max(np.abs(fy)))
        if max_force > 1.0:
            dt *= dt_decay
            dt = max(dt, dt_min)

        # Update positions
        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        # Clamp positions
        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

    return z


def _polish_result(z, time_budget):
    """Polish a valid packing with force-directed relaxation and adaptive radius growth."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dt = 0.003
    max_iter = int(min(1200, time_budget * 80))

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        for i in range(n):
            if x[i] - r[i] < 0:
                fx[i] += (x[i] - r[i]) * 40
            if x[i] + r[i] > w:
                fx[i] -= (x[i] + r[i] - w) * 40
            if y[i] - r[i] < 0:
                fy[i] += (y[i] - r[i]) * 40
            if y[i] + r[i] > h:
                fy[i] -= (y[i] + r[i] - h) * 40

        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-12:
                    overlap = min_d - d
                    fx[i] += (dx / d) * overlap * 20
                    fy[i] += (dy / d) * overlap * 20
                    fx[j] -= (dx / d) * overlap * 20
                    fy[j] -= (dy / d) * overlap * 20

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Adaptive radius growth: try growing all radii, then individually
    # First: uniform growth
    grow_factor = 1.001
    for _ in range(200):
        if time.time() > time_budget:
            break
        r_trial = r * grow_factor
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    # Second: try growing each circle individually
    for _ in range(50):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            r_trial = r.copy()
            r_trial[i] *= 1.002
            if _is_valid(x, y, r_trial, w, h):
                r = r_trial
                improved = True
        if not improved:
            break

    # Third: try growing all again after individual
    grow_factor = 1.0005
    for _ in range(100):
        if time.time() > time_budget:
            break
        r_trial = r * grow_factor
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    # Reconstruct z
    z_new = np.empty(3 * N + 1)
    z_new[0::3][:n] = x
    z_new[1::3][:n] = y
    z_new[2::3][:n] = r
    z_new[-1] = w

    if _is_valid(x, y, r, w, h):
        return z_new
    return None


def _constraint_refine(z, time_budget):
    """Constraint-based SLSQP refinement: maximize sum(r) subject to explicit constraints."""
    z = z.copy().astype(float)
    n = N
    x0 = z[0::3][:n].copy()
    y0 = z[1::3][:n].copy()
    r0 = z[2::3][:n].copy()
    w0 = z[-1]
    h0 = 2.0 - w0

    def obj(z):
        r = z[2::3][:n]
        return -np.sum(r)

    # Build explicit constraints
    constraints = []

    # Non-negativity of radii handled by bounds
    # Overlap constraints: d_ij >= r_i + r_j  =>  d_ij - r_i - r_j >= 0
    # We use smooth form: (d_ij)^2 - (r_i + r_j)^2 >= 0  (squared to keep smooth)
    # But this is non-smooth at d=0. Use: d_ij - r_i - r_j >= 0 with sqrt handled carefully.
    # Better: use the penalty-free form with auxiliary smooth approximation.
    # For SLSQP, we provide inequality constraints g(z) >= 0.
    # d_ij - r_i - r_j >= 0 where d_ij = sqrt((xi-xj)^2 + (yi-yj)^2)
    # This is smooth except when xi=xj and yi=yj.
    # We'll provide them as separate constraint functions.

    # To limit number of constraints, only add those that are "active" or near-active
    # at the current solution
    for i in range(n):
        for j in range(i + 1, n):
            dx0 = x0[i] - x0[j]
            dy0 = y0[i] - y0[j]
            d0 = np.sqrt(dx0 * dx0 + dy0 * dy0)
            rs0 = r0[i] + r0[j]
            # Only add constraint if circles are close (within 2x their combined radius)
            if d0 < 2.5 * rs0:
                def overlap_con(z, i=i, j=j):
                    dx = z[3*i] - z[3*j]
                    dy = z[3*i+1] - z[3*j+1]
                    d = np.sqrt(dx*dx + dy*dy)
                    return d - z[3*i+2] - z[3*j+2]
                constraints.append({'type': 'ineq', 'fun': overlap_con})

    # Boundary constraints: x_i >= r_i, x_i <= w - r_i, y_i >= r_i, y_i <= h - r_i
    for i in range(n):
        def left_wall(z, i=i):
            return z[3*i] - z[3*i+2]
        def right_wall(z, i=i):
            w = z[-1]
            return w - z[3*i] - z[3*i+2]
        def bottom_wall(z, i=i):
            return z[3*i+1] - z[3*i+2]
        def top_wall(z, i=i):
            h = 2.0 - z[-1]
            return h - z[3*i+1] - z[3*i+2]
        constraints.append({'type': 'ineq', 'fun': left_wall})
        constraints.append({'type': 'ineq', 'fun': right_wall})
        constraints.append({'type': 'ineq', 'fun': bottom_wall})
        constraints.append({'type': 'ineq', 'fun': top_wall})

    # Aspect ratio constraint: 0.4 <= w <= 1.6
    def w_lower(z):
        return z[-1] - 0.4
    def w_upper(z):
        return 1.6 - z[-1]
    constraints.append({'type': 'ineq', 'fun': w_lower})
    constraints.append({'type': 'ineq', 'fun': w_upper})

    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((max(0, x0[i] - 0.08), min(w0, x0[i] + 0.08)))
        bounds.append((max(0, y0[i] - 0.08), min(h0, y0[i] + 0.08)))
        bounds.append((max(1e-4, r0[i] * 0.95), r0[i] * 1.08))
    bounds.append((max(0.4, w0 - 0.05), min(1.6, w0 + 0.05)))

    maxiter = int(min(800, time_budget * 50))
    try:
        res = minimize(obj, z, method='SLSQP', bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 200})
        z_opt = res.x

        if _is_valid(z_opt[0::3][:n], z_opt[1::3][:n], z_opt[2::3][:n],
                     z_opt[-1], 2.0 - z_opt[-1]):
            return z_opt
    except Exception:
        pass

    return None


def _high_precision_refine(z, time_budget):
    """High-precision SLSQP refinement with tight tolerances and vectorized objective."""
    z = z.copy().astype(float)
    n = N
    x0 = z[0::3][:n].copy()
    y0 = z[1::3][:n].copy()
    r0 = z[2::3][:n].copy()
    w0 = z[-1]
    h0 = 2.0 - w0

    # Build vectorized objective for high-precision refinement
    penalty_weight = 2000.0

    def obj(z):
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w

        if w < 0.3 or h < 0.3:
            return 1e6

        val = -np.sum(r)

        # Vectorized overlap penalty
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx * dx + dy * dy)
        dist_req = r[:, None] + r[None, :]
        iu = np.triu_indices(n, 1)
        overlaps = np.maximum(0, dist_req[iu] - d[iu])
        val += penalty_weight * np.sum(overlaps ** 2)

        # Vectorized boundary penalty
        bx = np.maximum(0, r - x) + np.maximum(0, x - (w - r))
        by = np.maximum(0, r - y) + np.maximum(0, y - (h - r))
        val += penalty_weight * (np.sum(bx ** 2) + np.sum(by ** 2))

        return val

    # Tight bounds around current solution
    bounds = []
    for i in range(n):
        bounds.append((max(0, x0[i] - 0.03), min(w0, x0[i] + 0.03)))
        bounds.append((max(0, y0[i] - 0.03), min(h0, y0[i] + 0.03)))
        bounds.append((max(1e-4, r0[i] * 0.98), r0[i] * 1.03))
    bounds.append((max(0.3, w0 - 0.03), min(1.7, w0 + 0.03)))

    maxiter = int(min(500, time_budget * 40))
    try:
        res = minimize(obj, z, method='SLSQP', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})
        z_opt = res.x

        if _is_valid(z_opt[0::3][:n], z_opt[1::3][:n], z_opt[2::3][:n],
                     z_opt[-1], 2.0 - z_opt[-1]):
            return z_opt
    except Exception:
        pass

    return None


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.05:
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
