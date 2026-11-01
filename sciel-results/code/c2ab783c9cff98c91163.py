"""Circle packing in perimeter-4 rectangle: SLSQP + LSE Refine + Coordinate Descent."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


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


def _lse_refine(x, y, r, w, h, time_budget):
    """
    Log-sum-exp based refinement: jointly optimize positions and radii
    using a smooth surrogate for the nonsmooth packing constraints.
    """
    n = N
    # Parameters: x (n), y (n), r (n), w (1)
    z = np.concatenate([x, y, r, [w]])

    def objective(z):
        x_ = z[:n]
        y_ = z[n:2*n]
        r_ = z[2*n:3*n]
        w_ = z[3*n]
        h_ = 2.0 - w_

        if w_ < 0.2 or h_ < 0.2 or np.any(r_ < 0):
            return 1e6

        obj = -np.sum(r_)

        # Log-sum-exp overlap penalty
        dx = x_[:, None] - x_[None, :]
        dy = y_[:, None] - y_[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r_[:, None] + r_[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = r_sum_pairs - d_pairs

        temp = 0.005
        soft_overlap = temp * np.log1p(np.exp(overlap / temp))
        obj += 800.0 * np.sum(soft_overlap)

        # Boundary penalties
        left_viol = r_ - x_
        right_viol = x_ + r_ - w_
        bottom_viol = r_ - y_
        top_viol = y_ + r_ - h_

        soft_left = temp * np.log1p(np.exp(left_viol / temp))
        soft_right = temp * np.log1p(np.exp(right_viol / temp))
        soft_bottom = temp * np.log1p(np.exp(bottom_viol / temp))
        soft_top = temp * np.log1p(np.exp(top_viol / temp))

        obj += 800.0 * (np.sum(soft_left) + np.sum(soft_right) + np.sum(soft_bottom) + np.sum(soft_top))

        return obj

    bounds = []
    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    maxiter = int(min(1000, time_budget * 80))
    try:
        res = minimize(objective, z, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-15, 'maxls': 200})

        z_opt = res.x
        x_opt = z_opt[:n]
        y_opt = z_opt[n:2*n]
        r_opt = z_opt[2*n:3*n]
        w_opt = z_opt[3*n]
        h_opt = 2.0 - w_opt

        if not _is_valid(x_opt, y_opt, r_opt, w_opt, h_opt):
            for factor in [0.999, 0.998, 0.997, 0.995, 0.99]:
                r_trial = r_opt * factor
                if _is_valid(x_opt, y_opt, r_trial, w_opt, h_opt):
                    r_opt = r_trial
                    break

        if _is_valid(x_opt, y_opt, r_opt, w_opt, h_opt):
            return x_opt, y_opt, r_opt, w_opt
        return None
    except Exception:
        return None


def _constraint_refine(z, time_budget):
    """High-precision SLSQP refinement using explicit inequality constraints."""
    z = z.copy().astype(float)
    n = N

    def objective(z):
        r = z[2::3][:n]
        return -np.sum(r)

    def con_non_overlap(z):
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        iu = np.triu_indices(n, k=1)
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        d_sq = dx * dx + dy * dy
        r_sum = r[iu[0]] + r[iu[1]]
        return d_sq - r_sum * r_sum

    def con_boundary_left(z):
        x = z[0::3][:n]
        r = z[2::3][:n]
        return x - r

    def con_boundary_right(z):
        x = z[0::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        return w - x - r

    def con_boundary_bottom(z):
        y = z[1::3][:n]
        r = z[2::3][:n]
        return y - r

    def con_boundary_top(z):
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w
        return h - y - r

    constraints = [
        {'type': 'ineq', 'fun': con_non_overlap},
        {'type': 'ineq', 'fun': con_boundary_left},
        {'type': 'ineq', 'fun': con_boundary_right},
        {'type': 'ineq', 'fun': con_boundary_bottom},
        {'type': 'ineq', 'fun': con_boundary_top}
    ]

    bounds = []
    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    maxiter = int(min(1500, time_budget * 100))
    try:
        res = minimize(objective, z, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-16, 'maxls': 200})

        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w

        if not _is_valid(x, y, r, w, h):
            for factor in [0.999, 0.998, 0.997, 0.995, 0.99]:
                r_trial = r * factor
                if _is_valid(x, y, r_trial, w, h):
                    r = r_trial
                    break

        if _is_valid(x, y, r, w, h):
            z_opt[2::3][:n] = r
            return z_opt
        return None
    except Exception:
        return None


def _coordinate_descent_polish(x, y, r, w, h, time_budget):
    """
    Coordinate descent polish: iteratively optimize one variable at a time
    to escape local minima of gradient-based methods.
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    n = N
    best_sum = float(np.sum(r))

    for _ in range(200):
        if time.time() > time_budget:
            break

        improved = False

        # Optimize x positions
        for i in range(n):
            if time.time() > time_budget:
                break
            lo = max(r[i], x[i] - 0.02)
            hi = min(w - r[i], x[i] + 0.02)
            if lo >= hi:
                continue
            best_x = x[i]
            best_val = -float('inf')
            for trial_x in np.linspace(lo, hi, 11):
                x_trial = x.copy()
                x_trial[i] = trial_x
                if _is_valid(x_trial, y, r, w, h):
                    val = float(np.sum(r))
                    if val > best_val:
                        best_val = val
                        best_x = trial_x
            if best_x != x[i]:
                x[i] = best_x
                improved = True

        # Optimize y positions
        for i in range(n):
            if time.time() > time_budget:
                break
            lo = max(r[i], y[i] - 0.02)
            hi = min(h - r[i], y[i] + 0.02)
            if lo >= hi:
                continue
            best_y = y[i]
            best_val = -float('inf')
            for trial_y in np.linspace(lo, hi, 11):
                y_trial = y.copy()
                y_trial[i] = trial_y
                if _is_valid(x, y_trial, r, w, h):
                    val = float(np.sum(r))
                    if val > best_val:
                        best_val = val
                        best_y = trial_y
            if best_y != y[i]:
                y[i] = best_y
                improved = True

        # Optimize radii
        for i in range(n):
            if time.time() > time_budget:
                break
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
                lo = r[i]
                hi = r_max
                for _ in range(20):
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-9:
                    r[i] = lo
                    improved = True

        if not improved:
            break

    if _is_valid(x, y, r, w, h):
        return x, y, r, w
    return None


def _force_relax(x, y, r, w, h, max_iter=200):
    """Simple force-directed relaxation to resolve overlaps."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    n = N
    dt = 0.002

    for it in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        for i in range(n):
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * 50
            if x[i] > w - r[i]:
                fx[i] -= (x[i] - (w - r[i])) * 50
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * 50
            if y[i] > h - r[i]:
                fy[i] -= (y[i] - (h - r[i])) * 50

        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_sum = r[i] + r[j]
                if d < r_sum and d > 1e-10:
                    overlap = r_sum - d
                    fx[i] += (dx / d) * overlap * 10
                    fy[i] += (dy / d) * overlap * 10
                    fx[j] -= (dx / d) * overlap * 10
                    fy[j] -= (dy / d) * overlap * 10

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

        # Check convergence
        max_overlap = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                overlap = r[i] + r[j] - d
                if overlap > max_overlap:
                    max_overlap = overlap
        if max_overlap < 1e-8:
            break

    return x, y


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
    deadline = start_time + 140.0

    best_value = 0.0
    best_result = None

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

    # Generate diverse initial guesses
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

        # Random starts
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
        if time.time() > deadline - 25:
            break

        # Force-directed relaxation to resolve initial overlaps
        x0 = z0[0::3][:N]
        y0 = z0[1::3][:N]
        r0 = z0[2::3][:N]
        w0 = z0[-1]
        h0 = 2.0 - w0
        x_rel, y_rel = _force_relax(x0, y0, r0, w0, h0, max_iter=100)

        z0_relaxed = z0.copy()
        z0_relaxed[0::3][:N] = x_rel
        z0_relaxed[1::3][:N] = y_rel

        # SLSQP optimization
        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(500, max(100, remaining * 40)))
        try:
            res = minimize(_vectorized_objective, z0_relaxed, method='SLSQP',
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

    # --- Phase 2: Advanced Polish on best result ---
    if best_result is not None and time.time() < deadline - 35:
        # 1. LSE Refine
        x = best_result[0::3][:N]
        y = best_result[1::3][:N]
        r = best_result[2::3][:N]
        w = best_result[-1]
        h = 2.0 - w
        result = _lse_refine(x, y, r, w, h, deadline - (time.time() - start_time) - 25)
        if result is not None:
            x_lse, y_lse, r_lse, w_lse = result
            valid, val = check_validity(np.concatenate([x_lse, y_lse, r_lse, [w_lse]]))
            if valid and val > best_value:
                best_value = val
                best_result = np.concatenate([x_lse, y_lse, r_lse, [w_lse]])

        # 2. Constraint-based SLSQP refinement
        if best_result is not None and time.time() < deadline - 25:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 15)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # 3. Coordinate Descent Polish
        if best_result is not None and time.time() < deadline - 15:
            x = best_result[0::3][:N]
            y = best_result[1::3][:N]
            r = best_result[2::3][:N]
            w = best_result[-1]
            h = 2.0 - w
            result = _coordinate_descent_polish(x, y, r, w, h, deadline - (time.time() - start_time) - 10)
            if result is not None:
                x_cd, y_cd, r_cd, w_cd = result
                valid, val = check_validity(np.concatenate([x_cd, y_cd, r_cd, [w_cd]]))
                if valid and val > best_value:
                    best_value = val
                    best_result = np.concatenate([x_cd, y_cd, r_cd, [w_cd]])

        # 4. Second LSE Refine pass
        if best_result is not None and time.time() < deadline - 10:
            x = best_result[0::3][:N]
            y = best_result[1::3][:N]
            r = best_result[2::3][:N]
            w = best_result[-1]
            h = 2.0 - w
            result = _lse_refine(x, y, r, w, h, deadline - (time.time() - start_time) - 5)
            if result is not None:
                x_lse, y_lse, r_lse, w_lse = result
                valid, val = check_validity(np.concatenate([x_lse, y_lse, r_lse, [w_lse]]))
                if valid and val > best_value:
                    best_value = val
                    best_result = np.concatenate([x_lse, y_lse, r_lse, [w_lse]])

        # 5. Final Constraint Refine
        if best_result is not None and time.time() < deadline - 5:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time))
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
    w = best_result[-1]
    h = 2.0 - w

    # Final safety: ensure strict validity with minimal shrink
    for _ in range(20):
        r_trial = r * (1 - 1e-8)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
