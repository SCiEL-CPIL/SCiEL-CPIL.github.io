"""Circle packing in perimeter-4 rectangle: combined force-relax + SLSQP + constraint polish."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21

# EVOLVE-BLOCK-START

def _vectorized_objective(z, penalty_weight=2000.0):
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


def _force_relax(z, max_iter=800, deadline=None):
    """Force-directed relaxation to resolve overlaps and boundary violations."""
    z = z.copy().astype(float)
    dt = 0.003
    min_dt = 1e-5
    convergence_tol = 1e-10

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w < 0.1 or h < 0.1:
            break

        fx = np.zeros(N)
        fy = np.zeros(N)

        # Wall repulsion
        for i in range(N):
            if x[i] - r[i] < 0:
                fx[i] += (x[i] - r[i]) * 50.0
            if x[i] + r[i] - w > 0:
                fx[i] -= (x[i] + r[i] - w) * 50.0
            if y[i] - r[i] < 0:
                fy[i] += (y[i] - r[i]) * 50.0
            if y[i] + r[i] - h > 0:
                fy[i] -= (y[i] + r[i] - h) * 50.0

        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy) + 1e-12
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    force = (min_dist - dist) * 30.0 / dist
                    fx[i] += dx * force
                    fy[i] += dy * force
                    fx[j] -= dx * force
                    fy[j] -= dy * force

        # Update positions
        z[0::3][:N] += dt * fx
        z[1::3][:N] += dt * fy

        # Clamp to rectangle
        for i in range(N):
            z[3*i] = np.clip(z[3*i], 0.001, w - 0.001)
            z[3*i+1] = np.clip(z[3*i+1], 0.001, h - 0.001)

        # Check convergence
        if np.max(np.abs(fx)) + np.max(np.abs(fy)) < convergence_tol:
            break

        # Decay step size
        dt = max(dt * 0.995, min_dt)

    return z


def _check_validity(z, tol=1e-8):
    """Check if configuration is valid. Returns (valid, value)."""
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


def _constraint_refine(z, deadline):
    """SLSQP with explicit constraints for high-precision refinement."""
    x0 = z[0::3][:N]
    y0 = z[1::3][:N]
    r0 = z[2::3][:N]
    w0 = z[-1]
    h0 = 2.0 - w0

    def objective(p):
        x = p[0:N]
        y = p[N:2*N]
        r = p[2*N:3*N]
        w = p[3*N]
        h = 2.0 - w
        if w < 0.1 or h < 0.1:
            return 1e6
        return -np.sum(r)

    # Inequality constraints: g(x) >= 0
    constraints = []

    # Boundary constraints
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[i] - p[2*N+i]})  # x >= r
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*N] - p[i] - p[2*N+i]})  # x+r <= w
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[N+i] - p[2*N+i]})  # y >= r
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: (2.0 - p[3*N]) - p[N+i] - p[2*N+i]})  # y+r <= h

    # Pairwise non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (p[i]-p[j])**2 + (p[N+i]-p[N+j])**2 - (p[2*N+i]+p[2*N+j])**2
            })

    # Radius positivity
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*N+i]})

    bounds = []
    for i in range(N):
        bounds.append((0.001, 1.0))  # x
    for i in range(N):
        bounds.append((0.001, 1.0))  # y
    for i in range(N):
        bounds.append((0.001, 0.5))  # r
    bounds.append((0.3, 1.7))  # w

    p0 = np.concatenate([x0, y0, r0, [w0]])

    try:
        res = minimize(objective, p0, method='SLSQP', bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': 200, 'ftol': 1e-14, 'maxls': 100})
        if res.success or (not res.success and res.fun < -np.sum(r0) - 1e-6):
            p_opt = res.x
            z_opt = np.concatenate([p_opt[:N], p_opt[N:2*N], p_opt[2*N:3*N], [p_opt[3*N]]])
            valid, val = _check_validity(z_opt)
            if valid and val > np.sum(r0):
                return z_opt
    except Exception:
        pass

    return None


def _polish_result(z, deadline):
    """Iterative force-relax + SLSQP polish for high-precision refinement."""
    current = z.copy()
    current_val = 0.0
    valid, current_val = _check_validity(current)
    if not valid:
        return None

    for iteration in range(5):
        if time.time() > deadline:
            break

        # Force relax to resolve any small violations
        relaxed = _force_relax(current, max_iter=300, deadline=deadline)
        valid, val = _check_validity(relaxed)
        if valid and val >= current_val - 1e-10:
            current = relaxed
            current_val = val

        # SLSQP refinement
        if time.time() > deadline - 10:
            break
        refined = _constraint_refine(current, deadline)
        if refined is not None:
            valid, val = _check_validity(refined)
            if valid and val > current_val:
                current = refined
                current_val = val
            else:
                break

    return current


def construct():
    """
    Pack 21 circles in a variable rectangle of perimeter 4.
    Returns array of shape (21, 3): [cx, cy, r] for each circle.
    Uses force-directed relaxation + SLSQP optimization with constraint-based polish.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None

    # Generate diverse initial guesses
    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))

        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8, 1.4, 0.7]
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
        for _ in range(15):
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

    # Phase 1: Main optimization loop
    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 15:
            break

        # Force-directed relaxation to resolve initial overlaps
        z0_relaxed = _force_relax(z0, max_iter=500, deadline=deadline)

        # SLSQP optimization
        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(500, max(100, remaining * 30)))
        try:
            res = minimize(_vectorized_objective, z0_relaxed, method='L-BFGS-B',
                           bounds=[(0.001, 1.0)] * (2 * N) + [(0.001, 0.5)] * N + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10})

            z_opt = res.x
            valid, val = _check_validity(z_opt)

            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
            elif not valid:
                # Try repair: shrink radii slightly
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.92
                valid, val = _check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair.copy()

        except Exception:
            continue

    # Phase 2: High-precision polish on best result
    if best_result is not None and time.time() < deadline - 20:
        polished = _polish_result(best_result, deadline - (time.time() - start_time))
        if polished is not None:
            valid, val = _check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

    # Phase 3: Additional high-precision polish pass
    if best_result is not None and time.time() < deadline - 10:
        polished2 = _polish_result(best_result, deadline - (time.time() - start_time))
        if polished2 is not None:
            valid, val = _check_validity(polished2)
            if valid and val > best_value:
                best_value = val
                best_result = polished2

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

    # Final safety: ensure strict validity with small shrink
    for _ in range(100):
        r_trial = r * (1 - 1e-8)
        if _check_validity(np.concatenate([x, y, r_trial, [w]]))[0]:
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))

# EVOLVE-BLOCK-END
