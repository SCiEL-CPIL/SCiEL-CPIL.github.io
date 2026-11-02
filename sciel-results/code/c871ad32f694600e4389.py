"""Circle packing in perimeter-4 rectangle: grid-cell init + smooth push-apart + SLSQP + expansion polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE_MARGIN = 10.0


def _radius_sum(circles):
    return float(np.sum(circles[:, 2]))


def _is_valid(circles, tol=1e-9):
    if len(circles) != N:
        return False
    if np.any(circles[:, 2] <= 0):
        return False
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]

    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    W = max_x - min_x
    H = max_y - min_y
    if W + H > 2.0 + tol:
        return False

    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx * dx + dy * dy
            r_sum = rs[i] + rs[j]
            if dist_sq < r_sum * r_sum - tol:
                return False
    return True


def _is_valid_from_z(z, tol=1e-9):
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w <= 0.05 or h <= 0.05:
        return False
    if np.any(r <= 0):
        return False

    for i in range(N):
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False

    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            r_sum = r[i] + r[j]
            if d2 < r_sum * r_sum - tol:
                return False
    return True


def _radius_sum_from_z(z):
    r = z[2::3][:N]
    return float(np.sum(r))


def _smooth_push_apart_energy(z, beta=20.0, boundary_weight=50.0):
    """
    Smooth energy using log-sum-exp for overlap and boundary violations.
    Lower energy = better packing.
    """
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.1 or h < 0.1:
        return 1e6

    # Pairwise overlap: violation_ij = (r_i + r_j)^2 - d_ij^2
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum_sq = (r[:, None] + r[None, :]) ** 2

    iu = np.triu_indices(N, k=1)
    violations = r_sum_sq[iu] - d_sq[iu]

    v_max = np.max(violations)
    if v_max > 0:
        smooth_overlap = (1.0 / beta) * np.log(np.sum(np.exp(beta * (violations - v_max))) + 1e-30) + v_max
    else:
        smooth_overlap = 0.0

    # Boundary violations
    left_viol = r - x
    right_viol = x + r - w
    bottom_viol = r - y
    top_viol = y + r - h

    all_bviol = np.concatenate([left_viol, right_viol, bottom_viol, top_viol])
    bv_max = np.max(all_bviol)
    if bv_max > 0:
        smooth_boundary = (1.0 / beta) * np.log(np.sum(np.exp(beta * (all_bviol - bv_max))) + 1e-30) + bv_max
    else:
        smooth_boundary = 0.0

    return smooth_overlap + boundary_weight * smooth_boundary


def _smooth_push_apart_gradient(z, beta=20.0, boundary_weight=50.0):
    """
    Compute gradient of the smooth push-apart energy.
    """
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.1 or h < 0.1:
        return np.zeros_like(z)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    r_sum_sq = r_sum ** 2

    iu = np.triu_indices(N, k=1)
    violations = r_sum_sq[iu] - d_sq[iu]
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))

    v_max = np.max(violations)
    if v_max > 0:
        weights = np.exp(beta * (violations - v_max))
        weights /= np.sum(weights) + 1e-30
    else:
        weights = np.zeros(len(iu[0]))

    grad_x = np.zeros(N)
    grad_y = np.zeros(N)
    grad_r = np.zeros(N)

    for idx in range(len(iu[0])):
        i = iu[0][idx]
        j = iu[1][idx]
        wt = weights[idx]
        if wt < 1e-10:
            continue
        d = d_pairs[idx]
        if d < 1e-10:
            continue
        grad_x[i] -= 2.0 * wt * (x[i] - x[j])
        grad_x[j] += 2.0 * wt * (x[i] - x[j])
        grad_y[i] -= 2.0 * wt * (y[i] - y[j])
        grad_y[j] += 2.0 * wt * (y[i] - y[j])
        grad_r[i] += 2.0 * wt * r_sum[idx]
        grad_r[j] += 2.0 * wt * r_sum[idx]

    left_viol = r - x
    right_viol = x + r - w
    bottom_viol = r - y
    top_viol = y + r - h

    all_bviol = np.concatenate([left_viol, right_viol, bottom_viol, top_viol])
    bv_max = np.max(all_bviol)
    if bv_max > 0:
        bweights = np.exp(beta * (all_bviol - bv_max))
        bweights /= np.sum(bweights) + 1e-30
    else:
        bweights = np.zeros(4 * N)

    lw = bweights[:N]
    grad_x -= boundary_weight * lw
    grad_r += boundary_weight * lw

    rw = bweights[N:2*N]
    grad_x += boundary_weight * rw
    grad_r += boundary_weight * rw

    bw = bweights[2*N:3*N]
    grad_y -= boundary_weight * bw
    grad_r += boundary_weight * bw

    tw = bweights[3*N:4*N]
    grad_y += boundary_weight * tw
    grad_r += boundary_weight * tw

    grad = np.zeros_like(z)
    grad[0::3][:N] = grad_x
    grad[1::3][:N] = grad_y
    grad[2::3][:N] = grad_r
    grad[-1] = -boundary_weight * (np.sum(rw) + np.sum(tw))

    return grad


def _gradient_descent_flow(z, time_budget=15.0, beta=20.0, lr_init=0.005):
    """
    Gradient descent on smooth push-apart energy with adaptive learning rate.
    """
    z = z.copy().astype(float)
    start = time.time()
    lr = lr_init
    best_energy = _smooth_push_apart_energy(z, beta)
    best_z = z.copy()
    patience = 0

    for it in range(2000):
        if time.time() - start > time_budget:
            break

        grad = _smooth_push_apart_gradient(z, beta)
        z_new = z - lr * grad

        for i in range(N):
            r_i = z_new[2::3][:N][i]
            w = z_new[-1]
            h = 2.0 - w
            z_new[3*i] = np.clip(z_new[3*i], 0.01, max(w - 0.01, 0.02))
            z_new[3*i+1] = np.clip(z_new[3*i+1], 0.01, max(h - 0.01, 0.02))
            z_new[3*i+2] = np.clip(z_new[3*i+2], 0.001, 0.5)
        z_new[-1] = np.clip(z_new[-1], 0.3, 1.7)

        energy = _smooth_push_apart_energy(z_new, beta)

        if energy < best_energy - 1e-8:
            best_energy = energy
            best_z = z_new.copy()
            patience = 0
            lr = min(lr * 1.05, 0.02)
        else:
            patience += 1
            lr *= 0.95
            if patience > 50:
                break

    return best_z


def _force_relax(z, max_iter=500, dt_init=0.003):
    """Force-directed relaxation to resolve overlaps and push circles into bounds."""
    z = z.copy().astype(float)
    dt = dt_init
    for it in range(max_iter):
        x = z[0::3][:N].copy()
        y = z[1::3][:N].copy()
        r = z[2::3][:N].copy()
        w = z[-1]
        h = 2.0 - w

        fx = np.zeros(N)
        fy = np.zeros(N)

        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_sum = r[i] + r[j]
                if d < r_sum:
                    if d < 1e-12:
                        dx = 0.01
                        dy = 0.01
                        d = np.sqrt(dx * dx + dy * dy)
                    force = (r_sum - d) * 10.0
                    fx[i] += force * dx / d
                    fy[i] += force * dy / d
                    fx[j] -= force * dx / d
                    fy[j] -= force * dy / d

        for i in range(N):
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * 50.0
            if x[i] > w - r[i]:
                fx[i] -= (x[i] - (w - r[i])) * 50.0
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * 50.0
            if y[i] > h - r[i]:
                fy[i] -= (y[i] - (h - r[i])) * 50.0

        z[0::3][:N] += dt * fx
        z[1::3][:N] += dt * fy

        for i in range(N):
            z[3 * i] = np.clip(z[3 * i], r[i] * 0.5, w - r[i] * 0.5)
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i] * 0.5, h - r[i] * 0.5)

        dt *= 0.995
        if dt < 1e-6:
            break

    return z


def _grid_cell_layout(w, h, n_cols, n_rows, margin=0.95):
    """
    Grid-based cellular decomposition.
    Returns z vector of shape (3N + 1,).
    """
    total_cells = n_cols * n_rows
    if total_cells < N:
        return None

    z = np.zeros(3 * N + 1)
    cell_w = w / n_cols
    cell_h = h / n_rows

    count = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if count >= N:
                break
            cx = (col + 0.5) * cell_w
            cy = (row + 0.5) * cell_h
            r = 0.5 * min(cell_w, cell_h) * margin
            z[3 * count] = cx
            z[3 * count + 1] = cy
            z[3 * count + 2] = r
            count += 1
    z[-1] = w
    return z


def _vectorized_objective(z, penalty_weight=1000.0):
    """Vectorized objective: minimize -sum(r) + penalties."""
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
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
    r_pairs = r_sum[iu]
    overlap = np.maximum(r_pairs - d_pairs, 0.0)
    obj += penalty_weight * np.sum(overlap * overlap)

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

    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def _slsqp_refine(z0, time_budget=15.0):
    """SLSQP refinement on the unified parameter vector."""
    start = time.time()
    remaining = time_budget - (time.time() - start)
    maxiter = int(min(500, max(100, remaining * 40)))

    try:
        res = minimize(
            _vectorized_objective, z0, method='SLSQP',
            bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
            options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 60}
        )
        if res.success or res.fun < _vectorized_objective(z0):
            return res.x
    except Exception:
        pass
    return z0


def _slsqp_with_constraints(z0, time_budget=15.0):
    """SLSQP with explicit inequality constraints for tighter convergence."""
    start = time.time()

    def objective(zv):
        return -np.sum(zv[2::3][:N])

    constraints = []
    bounds = []

    for i in range(N):
        bounds.extend([(0.001, 2.0), (0.001, 2.0), (0.001, 0.5)])
    bounds.append((0.3, 1.7))

    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[-1] - p[3*i] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: (2.0 - p[-1]) - p[3*i+1] - p[3*i+2]})

    for i in range(N):
        for j in range(i + 1, N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (p[3*i] - p[3*j])**2 + (p[3*i+1] - p[3*j+1])**2 - (p[3*i+2] + p[3*j+2])**2
            })

    try:
        res = minimize(objective, z0, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': 300, 'ftol': 1e-12, 'maxls': 100})
        if res.success or res.fun < objective(z0):
            return res.x
    except Exception:
        pass
    return z0


def _expansion_polish(z, time_budget=8.0):
    """
    Try to expand radii while maintaining validity.
    Uses iterative radius scaling with force relaxation.
    """
    z = z.copy().astype(float)
    start = time.time()
    scale = 1.0
    best_z = z.copy()
    best_val = _radius_sum_from_z(z)

    for step in range(100):
        if time.time() - start > time_budget:
            break
        scale *= 1.008
        z_trial = z.copy()
        z_trial[2::3][:N] = z[2::3][:N] * scale

        if _is_valid_from_z(z_trial):
            z = z_trial
            val = _radius_sum_from_z(z)
            if val > best_val:
                best_val = val
                best_z = z.copy()
        else:
            z_relaxed = _force_relax(z_trial, max_iter=80, dt_init=0.002)
            if _is_valid_from_z(z_relaxed):
                z = z_relaxed
                val = _radius_sum_from_z(z)
                if val > best_val:
                    best_val = val
                    best_z = z.copy()
            else:
                break

    return best_z


def _socp_polish(z, time_budget=10.0):
    """
    High-precision polish using SLSQP with tighter tolerances and more iterations.
    This acts as a local exact solve to squeeze out remaining gains.
    """
    start = time.time()
    z = z.copy().astype(float)
    
    # Use the vectorized objective but with a very high penalty to enforce constraints strictly
    # and a tighter ftol
    def strict_objective(zv):
        return _vectorized_objective(zv, penalty_weight=5000.0)
    
    def strict_grad(zv):
        # Finite difference gradient for strict objective
        eps = 1e-7
        g = np.zeros_like(zv)
        f0 = strict_objective(zv)
        for i in range(len(zv)):
            zv_p = zv.copy()
            zv_p[i] += eps
            g[i] = (strict_objective(zv_p) - f0) / eps
        return g

    try:
        res = minimize(
            strict_objective, z, method='SLSQP',
            bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
            options={'maxiter': 800, 'ftol': 1e-14, 'maxls': 100}
        )
        if res.fun < strict_objective(z):
            if _is_valid_from_z(res.x):
                return res.x
            else:
                # Try repair
                z_repair = res.x.copy()
                z_repair[2::3][:N] *= 0.98
                if _is_valid_from_z(z_repair):
                    return z_repair
    except Exception:
        pass
    
    return z


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_z = None
    best_val = 0.0

    # --- Phase 1: Grid-cell decompositions with diverse aspect ratios ---
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.2, 0.8, 1.4, 0.7, 1.1, 0.9]

    for w0 in aspect_ratios:
        if time.time() > deadline - 60:
            break
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue

        for n_cols in range(3, 7):
            for n_rows in range(3, 7):
                if n_cols * n_rows < N:
                    continue
                z = _grid_cell_layout(w0, h0, n_cols, n_rows, margin=0.92)
                if z is None:
                    continue

                # Force relax to settle
                z_relaxed = _force_relax(z, max_iter=200, dt_init=0.003)

                if not _is_valid_from_z(z_relaxed):
                    continue

                val = _radius_sum_from_z(z_relaxed)
                if val > best_val:
                    best_val = val
                    best_z = z_relaxed.copy()

    # --- Phase 2: Smooth push-apart gradient descent ---
    if best_z is not None and time.time() < deadline - 40:
        smoothed = _gradient_descent_flow(best_z, time_budget=12.0, beta=20.0)
        if _is_valid_from_z(smoothed):
            val = _radius_sum_from_z(smoothed)
            if val > best_val:
                best_val = val
                best_z = smoothed.copy()

    # --- Phase 3: SLSQP with explicit constraints ---
    if best_z is not None and time.time() < deadline - 30:
        refined = _slsqp_with_constraints(best_z, time_budget=15.0)
        if _is_valid_from_z(refined):
            val = _radius_sum_from_z(refined)
            if val > best_val:
                best_val = val
                best_z = refined
        else:
            z_repair = refined.copy()
            z_repair[2::3][:N] *= 0.95
            if _is_valid_from_z(z_repair):
                val = _radius_sum_from_z(z_repair)
                if val > best_val:
                    best_val = val
                    best_z = z_repair

    # --- Phase 4: Expansion polish ---
    if best_z is not None and time.time() < deadline - 20:
        expanded = _expansion_polish(best_z, time_budget=10.0)
        if _is_valid_from_z(expanded):
            val = _radius_sum_from_z(expanded)
            if val > best_val:
                best_val = val
                best_z = expanded

    # --- Phase 5: SOCP-style high-precision polish ---
    if best_z is not None and time.time() < deadline - 10:
        polished = _socp_polish(best_z, time_budget=8.0)
        if _is_valid_from_z(polished):
            val = _radius_sum_from_z(polished)
            if val > best_val:
                best_val = val
                best_z = polished

    # --- Phase 6: Final SLSQP pass ---
    if best_z is not None and time.time() < deadline - 5:
        refined2 = _slsqp_refine(best_z, time_budget=4.0)
        if _is_valid_from_z(refined2):
            val = _radius_sum_from_z(refined2)
            if val > best_val:
                best_val = val
                best_z = refined2
        else:
            z_repair = refined2.copy()
            z_repair[2::3][:N] *= 0.96
            if _is_valid_from_z(z_repair):
                val = _radius_sum_from_z(z_repair)
                if val > best_val:
                    best_val = val
                    best_z = z_repair

    # --- Phase 7: Final force polish ---
    if best_z is not None and time.time() < deadline - 2:
        polished = _force_relax(best_z, max_iter=100, dt_init=0.001)
        if _is_valid_from_z(polished):
            val = _radius_sum_from_z(polished)
            if val > best_val:
                best_val = val
                best_z = polished

    # Fallback
    if best_z is None:
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side * 0.95
                cells.append([x, y, r])
        return np.array(cells[:N])

    # Format output
    x = best_z[0::3][:N]
    y = best_z[1::3][:N]
    r = best_z[2::3][:N]

    # Safety shrink
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        circles_trial = np.column_stack((x, y, r_trial))
        if _is_valid(circles_trial):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
