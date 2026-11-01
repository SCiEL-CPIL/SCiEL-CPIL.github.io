"""Circle packing in perimeter-4 rectangle: SLSQP + Contact Graph + LSE Refine + Greedy Expansion."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


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

    # Vectorized pairwise overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 0.0))
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    overlap_penalty = penalty_weight * np.sum(overlap * overlap)

    obj += overlap_penalty

    # Vectorized boundary penalty
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

    # Keep aspect ratio in reasonable range
    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

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
        grid_configs = [
            (5, 5), (4, 6), (6, 4), (3, 7), (7, 3), (2, 11), (11, 2)
        ]

        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            for rows, cols in grid_configs:
                z = np.zeros(3 * N + 1)
                count = 0
                for row in range(rows):
                    for col in range(cols):
                        if count >= N:
                            break
                        cx = (col + 0.5) / cols * w0
                        cy = (row + 0.5) / rows * h0
                        r0 = min(w0 / cols, h0 / rows) * 0.45
                        z[3 * count] = cx
                        z[3 * count + 1] = cy
                        z[3 * count + 2] = r0
                        count += 1
                z[-1] = w0
                guesses.append(z)

        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(8):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.04, 0.10)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        return guesses

    guesses = get_initial_guesses()

    # Sort by initial objective
    def _initial_score(z):
        try:
            return _vectorized_objective(z, penalty_weight)
        except Exception:
            return 1e6

    guesses.sort(key=_initial_score)

    # Phase 1: SLSQP multi-start
    max_starts = min(len(guesses), 12)
    for i in range(max_starts):
        if time.time() > deadline - 30:
            break

        z0 = guesses[i]
        z0_relaxed = _force_relax_unified(z0, max_iter=400)

        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(500, max(100, remaining * 30)))
        try:
            res = minimize(objective, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 60})

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

    # Phase 2: Polish pipeline
    if best_result is not None and time.time() < deadline - 20:
        # 1. Force-directed position polish
        polished = _polish_result(best_result, deadline - (time.time() - start_time) - 15)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        # 2. Contact-graph optimization (exploits active constraint structure)
        if best_result is not None and time.time() < deadline - 18:
            cg = _contact_graph_optimize(best_result, deadline - (time.time() - start_time) - 12)
            if cg is not None:
                valid, val = check_validity(cg)
                if valid and val > best_value:
                    best_value = val
                    best_result = cg

        # 3. LSE epigraph refinement (smooth surrogate with adaptive temperature)
        if best_result is not None and time.time() < deadline - 15:
            lse = _lse_refine(best_result, deadline - (time.time() - start_time) - 10)
            if lse is not None:
                valid, val = check_validity(lse)
                if valid and val > best_value:
                    best_value = val
                    best_result = lse

        # 4. Per-circle greedy expansion
        if best_result is not None and time.time() < deadline - 10:
            expanded = _greedy_expansion(best_result, deadline - (time.time() - start_time) - 6)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # 5. Final high-precision polish
        if best_result is not None and time.time() < deadline - 5:
            final = _final_precision_polish(best_result, deadline - (time.time() - start_time) - 2)
            if final is not None:
                valid, val = check_validity(final)
                if valid and val > best_value:
                    best_value = val
                    best_result = final

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Format output
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    # Final safety: ensure strict validity with minimal shrink
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))


def _force_relax_unified(z, max_iter=400):
    """Force-directed relaxation in unified parameter space."""
    z = z.copy().astype(float)
    n = N
    dt = 0.005
    min_dt = 0.0001
    iu = np.triu_indices(n, k=1)

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
        fx -= left_viol * 100
        fx += right_viol * 100
        fy -= bottom_viol * 100
        fy += top_viol * 100

        # Pairwise repulsion
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        max_overlap = float(np.max(overlap)) if overlap.size > 0 else 0.0

        if max_overlap > 1e-8:
            f_pairs = overlap * 15.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            np.add.at(fx, iu[0], ux * f_pairs)
            np.add.at(fy, iu[0], uy * f_pairs)
            np.add.at(fx, iu[1], -ux * f_pairs)
            np.add.at(fy, iu[1], -uy * f_pairs)
        else:
            break

        force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
        if force_mag > 1.0:
            dt = max(min_dt, dt * 0.5)
        elif force_mag < 0.1:
            dt = min(0.01, dt * 1.1)

        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy
        z[0::3][:n] = np.clip(z[0::3][:n], r, w - r)
        z[1::3][:n] = np.clip(z[1::3][:n], r, h - r)

    return z


def _polish_result(z, time_budget):
    """Polish a valid packing with force-directed position adjustment."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w
    iu = np.triu_indices(n, k=1)

    dt = 0.001
    max_iter = int(min(2000, time_budget * 100))

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 35
        fx += right_viol * 35
        fy -= bottom_viol * 35
        fy += top_viol * 35

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) > 1e-10:
            f_pairs = overlap * 18.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            np.add.at(fx, iu[0], ux * f_pairs)
            np.add.at(fy, iu[0], uy * f_pairs)
            np.add.at(fx, iu[1], -ux * f_pairs)
            np.add.at(fy, iu[1], -uy * f_pairs)

        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 0.1:
            dt = 0.0005
        else:
            dt = 0.001

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Binary search for max uniform scale
    lo_scale = 1.0
    hi_scale = 1.03
    for _ in range(30):
        if time.time() > time_budget:
            break
        mid_scale = (lo_scale + hi_scale) / 2.0
        r_trial = r * mid_scale
        if _is_valid_fast(x, y, r_trial, w, h):
            lo_scale = mid_scale
        else:
            hi_scale = mid_scale
    r = r * lo_scale

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _contact_graph_optimize(z, time_budget):
    """
    Contact-graph-based optimization: identify active contacts (circles touching
    each other or walls), then solve a reduced SLSQP problem with equality-like
    constraints only for active contacts. This gives a better-conditioned problem.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w
    iu = np.triu_indices(n, k=1)

    # Identify active contacts
    contact_tol = 1e-4
    pair_contacts = np.zeros((n, n), dtype=bool)
    wall_contacts = np.zeros((n, 4), dtype=bool)  # left, right, bottom, top

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(np.maximum(dx * dx + dy * dy, 1e-12))
    r_sum = r[:, None] + r[None, :]

    for i in range(n):
        for j in range(i + 1, n):
            if abs(d[i, j] - r_sum[i, j]) < contact_tol:
                pair_contacts[i, j] = True
                pair_contacts[j, i] = True
        if abs(x[i] - r[i]) < contact_tol:
            wall_contacts[i, 0] = True
        if abs(x[i] + r[i] - w) < contact_tol:
            wall_contacts[i, 1] = True
        if abs(y[i] - r[i]) < contact_tol:
            wall_contacts[i, 2] = True
        if abs(y[i] + r[i] - h) < contact_tol:
            wall_contacts[i, 3] = True

    # Build constraint set for active contacts
    constraints = []
    bounds = [(0, 2)] * (3 * n) + [(0.3, 1.7)]

    # Wall constraints for active wall contacts
    for i in range(n):
        if wall_contacts[i, 0]:  # left
            constraints.append({'type': 'eq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})
        if wall_contacts[i, 1]:  # right
            constraints.append({'type': 'eq', 'fun': lambda p, i=i: p[-1] - p[3*i] - p[3*i+2]})
        if wall_contacts[i, 2]:  # bottom
            constraints.append({'type': 'eq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})
        if wall_contacts[i, 3]:  # top
            constraints.append({'type': 'eq', 'fun': lambda p, i=i: (2.0 - p[-1]) - p[3*i+1] - p[3*i+2]})

    # Pairwise contact constraints
    for i in range(n):
        for j in range(i + 1, n):
            if pair_contacts[i, j]:
                constraints.append({
                    'type': 'eq',
                    'fun': lambda p, i=i, j=j: (
                        (p[3*i] - p[3*j])**2 +
                        (p[3*i+1] - p[3*j+1])**2 -
                        (p[3*i+2] + p[3*j+2])**2
                    )
                })

    # Inequality constraints for all non-contact pairs (must be separated)
    for i in range(n):
        for j in range(i + 1, n):
            if not pair_contacts[i, j]:
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda p, i=i, j=j: (
                        (p[3*i] - p[3*j])**2 +
                        (p[3*i+1] - p[3*j+1])**2 -
                        (p[3*i+2] + p[3*j+2])**2
                    )
                })

    # Inequality constraints for non-active walls
    for i in range(n):
        if not wall_contacts[i, 0]:
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})
        if not wall_contacts[i, 1]:
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[-1] - p[3*i] - p[3*i+2]})
        if not wall_contacts[i, 2]:
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})
        if not wall_contacts[i, 3]:
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: (2.0 - p[-1]) - p[3*i+1] - p[3*i+2]})

    def obj_cg(p):
        r_p = p[2::3][:n]
        return -np.sum(r_p)

    maxiter = int(min(600, time_budget * 80))
    try:
        res = minimize(obj_cg, z, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})

        z_opt = res.x
        if _is_valid(z_opt[0::3][:n], z_opt[1::3][:n], z_opt[2::3][:n], z_opt[-1], 2.0 - z_opt[-1]):
            return z_opt
    except Exception:
        pass

    # Fallback: try with relaxed equality constraints (ineq with small tolerance)
    constraints2 = []
    for i in range(n):
        if wall_contacts[i, 0]:
            constraints2.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2] + contact_tol})
            constraints2.append({'type': 'ineq', 'fun': lambda p, i=i: -(p[3*i] - p[3*i+2]) + contact_tol})
        if wall_contacts[i, 1]:
            constraints2.append({'type': 'ineq', 'fun': lambda p, i=i: p[-1] - p[3*i] - p[3*i+2] + contact_tol})
            constraints2.append({'type': 'ineq', 'fun': lambda p, i=i: -(p[-1] - p[3*i] - p[3*i+2]) + contact_tol})
        if wall_contacts[i, 2]:
            constraints2.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2] + contact_tol})
            constraints2.append({'type': 'ineq', 'fun': lambda p, i=i: -(p[3*i+1] - p[3*i+2]) + contact_tol})
        if wall_contacts[i, 3]:
            constraints2.append({'type': 'ineq', 'fun': lambda p, i=i: (2.0 - p[-1]) - p[3*i+1] - p[3*i+2] + contact_tol})
            constraints2.append({'type': 'ineq', 'fun': lambda p, i=i: -((2.0 - p[-1]) - p[3*i+1] - p[3*i+2]) + contact_tol})

    for i in range(n):
        for j in range(i + 1, n):
            if pair_contacts[i, j]:
                constraints2.append({'type': 'ineq', 'fun': lambda p, i=i, j=j:
                    (p[3*i] - p[3*j])**2 + (p[3*i+1] - p[3*j+1])**2 - (p[3*i+2] + p[3*j+2])**2 + contact_tol**2})
                constraints2.append({'type': 'ineq', 'fun': lambda p, i=i, j=j:
                    -((p[3*i] - p[3*j])**2 + (p[3*i+1] - p[3*j+1])**2 - (p[3*i+2] + p[3*j+2])**2) + contact_tol**2})

    try:
        res2 = minimize(obj_cg, z, method='SLSQP',
                        bounds=bounds,
                        constraints=constraints2,
                        options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})
        z_opt2 = res2.x
        if _is_valid(z_opt2[0::3][:n], z_opt2[1::3][:n], z_opt2[2::3][:n], z_opt2[-1], 2.0 - z_opt2[-1]):
            return z_opt2
    except Exception:
        pass

    return None


def _lse_refine(z, time_budget):
    """
    LSE (Log-Sum-Exp) epigraph refinement: use a smooth surrogate for the
    non-convex distance constraints with adaptive temperature that tightens
    as we converge.
    """
    z = z.copy().astype(float)
    n = N
    iu = np.triu_indices(n, k=1)

    # Adaptive temperature schedule
    k_values = [10.0, 20.0, 50.0, 100.0, 200.0]

    for k in k_values:
        if time.time() > time_budget:
            break

        def obj_lse(p, k_val=k):
            x_p = p[0::3][:n]
            y_p = p[1::3][:n]
            r_p = p[2::3][:n]
            w_p = p[-1]
            h_p = 2.0 - w_p

            if w_p < 0.2 or h_p < 0.2 or np.any(r_p < 0):
                return 1e6

            obj = -np.sum(r_p)

            # LSE penalty for pairwise overlaps
            dx = x_p[:, None] - x_p[None, :]
            dy = y_p[:, None] - y_p[None, :]
            d_sq = dx * dx + dy * dy
            r_sum = r_p[:, None] + r_p[None, :]

            d_sq_pairs = d_sq[iu]
            r_sum_pairs = r_sum[iu]
            d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
            viol_pairs = r_sum_pairs - d_pairs

            # Smooth max via LSE
            if np.max(viol_pairs) > 0:
                m = np.max(viol_pairs)
                lse_pairs = m + np.log(np.sum(np.exp(viol_pairs - m))) / k_val
            else:
                lse_pairs = 0.0
            obj += 500.0 * lse_pairs

            # LSE penalty for boundary violations
            left_viol = r_p - x_p
            right_viol = x_p + r_p - w_p
            bottom_viol = r_p - y_p
            top_viol = y_p + r_p - h_p
            all_viol = np.concatenate([left_viol, right_viol, bottom_viol, top_viol])

            if np.max(all_viol) > 0:
                m = np.max(all_viol)
                lse_bound = m + np.log(np.sum(np.exp(all_viol - m))) / k_val
            else:
                lse_bound = 0.0
            obj += 500.0 * lse_bound

            # Aspect ratio penalty
            if w_p < 0.5:
                obj += 100.0 * (0.5 - w_p) ** 2
            elif w_p > 1.5:
                obj += 100.0 * (w_p - 1.5) ** 2

            return obj

        maxiter = int(min(400, time_budget * 50))
        try:
            res = minimize(obj_lse, z, method='SLSQP',
                           bounds=[(0, 2)] * (3 * n) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 60})
            z_opt = res.x
            if _is_valid(z_opt[0::3][:n], z_opt[1::3][:n], z_opt[2::3][:n], z_opt[-1], 2.0 - z_opt[-1]):
                z = z_opt
        except Exception:
            continue

    # Final: try to grow radii
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    lo_scale = 1.0
    hi_scale = 1.02
    for _ in range(25):
        if time.time() > time_budget:
            break
        mid_scale = (lo_scale + hi_scale) / 2.0
        r_trial = r * mid_scale
        if _is_valid_fast(x, y, r_trial, w, h):
            lo_scale = mid_scale
        else:
            hi_scale = mid_scale
    r = r * lo_scale

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _greedy_expansion(z, time_budget):
    """
    Per-circle greedy expansion: iteratively expand each circle to its individual
    maximum radius (not just uniform scaling), capturing per-circle radius gains.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    max_rounds = 15
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break

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

            if r_max > r[i] + 1e-8:
                # Binary search for max valid radius
                lo = r[i]
                hi = r_max
                for _ in range(20):
                    if time.time() > time_budget:
                        break
                    mid = (lo + hi) / 2.0
                    r_trial = r.copy()
                    r_trial[i] = mid
                    if _is_valid_fast(x, y, r_trial, w, h, tol=1e-9):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-9:
                    r[i] = lo
                    improved = True

        if not improved:
            break

    # Final uniform scaling
    lo_scale = 1.0
    hi_scale = 1.01
    for _ in range(25):
        if time.time() > time_budget:
            break
        mid_scale = (lo_scale + hi_scale) / 2.0
        r_trial = r * mid_scale
        if _is_valid_fast(x, y, r_trial, w, h):
            lo_scale = mid_scale
        else:
            hi_scale = mid_scale
    r = r * lo_scale

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _final_precision_polish(z, time_budget):
    """High-precision final polish with alternating position optimization and radius maximization."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w
    iu = np.triu_indices(n, k=1)

    for round_num in range(10):
        if time.time() > time_budget:
            break

        # Position polish with small steps
        dt = 0.0002
        for _ in range(400):
            if time.time() > time_budget:
                break
            fx = np.zeros(n)
            fy = np.zeros(n)

            left_viol = np.maximum(r - x, 0.0)
            right_viol = np.maximum(x + r - w, 0.0)
            bottom_viol = np.maximum(r - y, 0.0)
            top_viol = np.maximum(y + r - h, 0.0)
            fx -= left_viol * 15
            fx += right_viol * 15
            fy -= bottom_viol * 15
            fy += top_viol * 15

            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d_sq = dx * dx + dy * dy
            r_sum = r[:, None] + r[None, :]
            d_sq_pairs = d_sq[iu]
            r_sum_pairs = r_sum[iu]
            d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
            overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

            if np.max(overlap) < 1e-14:
                break

            f_pairs = overlap * 6.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            np.add.at(fx, iu[0], ux * f_pairs)
            np.add.at(fy, iu[0], uy * f_pairs)
            np.add.at(fx, iu[1], -ux * f_pairs)
            np.add.at(fy, iu[1], -uy * f_pairs)

            x += dt * fx
            y += dt * fy
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)

        # Per-circle radius maximization
        r_wall = np.minimum(np.minimum(x, w - x), np.minimum(y, h - y))
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(np.maximum(dx * dx + dy * dy, 1e-12))
        r_allowed = d - r[None, :]
        np.fill_diagonal(r_allowed, np.inf)
        r_pair_min = np.min(r_allowed, axis=1)
        r_max = np.maximum(np.minimum(r_wall, r_pair_min), 0.0)

        for i in range(n):
            if time.time() > time_budget:
                break
            if r_max[i] > r[i] + 1e-12:
                lo = r[i]
                hi = r_max[i]
                for _ in range(25):
                    mid = (lo + hi) / 2.0
                    r_trial = r.copy()
                    r_trial[i] = mid
                    if _is_valid_fast(x, y, r_trial, w, h, tol=1e-10):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-13:
                    r[i] = lo

    # Final uniform scaling
    lo_scale = 1.0
    hi_scale = 1.005
    for _ in range(30):
        if time.time() > time_budget:
            break
        mid_scale = (lo_scale + hi_scale) / 2.0
        r_trial = r * mid_scale
        if _is_valid_fast(x, y, r_trial, w, h, tol=1e-10):
            lo_scale = mid_scale
        else:
            hi_scale = mid_scale
    r = r * lo_scale

    if _is_valid(x, y, r, w, h, tol=1e-10):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid - vectorized."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    if np.any(d_sq[iu] < r_sum[iu] ** 2 - tol):
        return False
    return True


def _is_valid_fast(cx, cy, r, W, H, tol=1e-7):
    """Fast validity check with looser tolerance for inner loops."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    if np.any(d_sq[iu] < r_sum[iu] ** 2 - tol):
        return False
    return True
