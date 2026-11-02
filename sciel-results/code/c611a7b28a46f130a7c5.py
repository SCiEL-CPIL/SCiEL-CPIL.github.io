"""Circle packing in perimeter-4 rectangle: SLSQP + Smooth Objective + Advanced Polish."""
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


def _max_radius_for_circle(x, y, r, w, h, i):
    """Compute the maximum radius for circle i given all other positions and radii."""
    r_wall = min(x[i], w - x[i], y[i], h - y[i])
    r_max = r_wall
    for j in range(N):
        if i == j:
            continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx * dx + dy * dy)
        r_allowed = d - r[j]
        if r_allowed < r_max:
            r_max = r_allowed
    return max(0.0, r_max)


def _high_precision_polish(x, y, r, w, h, time_budget):
    """
    High-precision final polish: iterative expansion with very tight tolerances,
    small position adjustments, simultaneous multi-circle moves, and uniform scaling.
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    n = N
    tol = 1e-13
    max_rounds = 400
    rng = np.random.default_rng(99)

    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        improved = False

        # Step 1: Expand each circle to its maximum possible radius
        for i in range(n):
            r_max = _max_radius_for_circle(x, y, r, w, h, i)
            if r_max > r[i] + tol:
                r[i] = r_max
                improved = True

        # Step 2: Try small position adjustments to enable further expansion
        if improved:
            continue

        # Step 2a: Single circle nudges (finer steps)
        for i in range(n):
            r_max_before = _max_radius_for_circle(x, y, r, w, h, i)
            if r_max_before <= r[i] + tol:
                continue

            for dx in [0.0003, -0.0003, 0.00015, -0.00015, 0.00008, -0.00008, 0.00004, -0.00004]:
                for dy in [0.0003, -0.0003, 0.00015, -0.00015, 0.00008, -0.00008, 0.00004, -0.00004]:
                    if dx == 0.0 and dy == 0.0:
                        continue
                    nx = x[i] + dx
                    ny = y[i] + dy

                    # Wall check
                    if nx < r[i] or nx > w - r[i] or ny < r[i] or ny > h - r[i]:
                        continue

                    # Neighbor check
                    valid = True
                    for j in range(n):
                        if i == j:
                            continue
                        ddx = nx - x[j]
                        ddy = ny - y[j]
                        d2 = ddx * ddx + ddy * ddy
                        rs = r[i] + r[j]
                        if d2 < rs * rs - tol:
                            valid = False
                            break

                    if valid:
                        x[i] = nx
                        y[i] = ny
                        r_new_max = _max_radius_for_circle(x, y, r, w, h, i)
                        if r_new_max > r[i] + tol:
                            r[i] = r_new_max
                            improved = True
                            break

        # Step 2b: Simultaneous 2-circle moves (random pairs)
        if not improved:
            for _attempt in range(20):
                i = rng.integers(0, n)
                j = rng.integers(0, n)
                if i == j:
                    continue
                dx1 = rng.normal(0, 0.0002)
                dy1 = rng.normal(0, 0.0002)
                dx2 = rng.normal(0, 0.0002)
                dy2 = rng.normal(0, 0.0002)

                nx1 = x[i] + dx1
                ny1 = y[i] + dy1
                nx2 = x[j] + dx2
                ny2 = y[j] + dy2

                # Wall checks
                if nx1 < r[i] or nx1 > w - r[i] or ny1 < r[i] or ny1 > h - r[i]:
                    continue
                if nx2 < r[j] or nx2 > w - r[j] or ny2 < r[j] or ny2 > h - r[j]:
                    continue

                # Pair i-j check
                ddx = nx1 - nx2
                ddy = ny1 - ny2
                if ddx * ddx + ddy * ddy < (r[i] + r[j]) ** 2 - tol:
                    continue

                # Check i against all others
                valid = True
                for k in range(n):
                    if k == i or k == j:
                        continue
                    ddx = nx1 - x[k]
                    ddy = ny1 - y[k]
                    if ddx * ddx + ddy * ddy < (r[i] + r[k]) ** 2 - tol:
                        valid = False
                        break
                if not valid:
                    continue

                # Check j against all others
                for k in range(n):
                    if k == i or k == j:
                        continue
                    ddx = nx2 - x[k]
                    ddy = ny2 - y[k]
                    if ddx * ddx + ddy * ddy < (r[j] + r[k]) ** 2 - tol:
                        valid = False
                        break
                if not valid:
                    continue

                # Apply and try to expand
                x[i] = nx1
                y[i] = ny1
                x[j] = nx2
                y[j] = ny2
                r_new_i = _max_radius_for_circle(x, y, r, w, h, i)
                r_new_j = _max_radius_for_circle(x, y, r, w, h, j)
                if r_new_i > r[i] + tol or r_new_j > r[j] + tol:
                    r[i] = r_new_i
                    r[j] = r_new_j
                    # Also try expanding all
                    for k in range(n):
                        r_max_k = _max_radius_for_circle(x, y, r, w, h, k)
                        if r_max_k > r[k] + tol:
                            r[k] = r_max_k
                    improved = True
                    break
                else:
                    # Revert
                    x[i] -= dx1
                    y[i] -= dy1
                    x[j] -= dx2
                    y[j] -= dy2

        # Step 3: Uniform scaling attempt with finer granularity
        if not improved:
            for factor in [1.0 + 3e-7, 1.0 + 1e-7, 1.0 + 5e-8, 1.0 + 2e-8, 1.0 + 1e-8, 1.0 + 5e-9]:
                r_trial = r * factor
                if _is_valid(x, y, r_trial, w, h, tol=1e-11):
                    r = r_trial
                    improved = True
                    break

        if not improved:
            break

    return x, y, r


def _constraint_refine(x, y, r, w, h, time_budget):
    """
    Constraint-based refinement: fix positions, solve for optimal radii via
    iterative max-radius computation with position relaxation.
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    n = N
    max_rounds = 500

    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        # Phase A: Maximize each radius given current positions
        changed = False
        for _pass in range(10):
            for i in range(n):
                r_max = _max_radius_for_circle(x, y, r, w, h, i)
                if r_max > r[i] + 1e-10:
                    r[i] = r_max
                    changed = True
            if not changed:
                break

        if not changed:
            break

        # Phase B: Try to improve by shifting circles that are constrained
        for i in range(n):
            r_max = _max_radius_for_circle(x, y, r, w, h, i)
            if r_max <= r[i] + 1e-8:
                continue

            # Find the most constraining neighbor or wall
            min_gap = float('inf')
            best_dx = 0.0
            best_dy = 0.0

            # Wall gaps
            gaps = [
                (x[i] - r[i], 1.0, 0.0),
                (w - x[i] - r[i], -1.0, 0.0),
                (y[i] - r[i], 0.0, 1.0),
                (h - y[i] - r[i], 0.0, -1.0),
            ]
            for gap, dx, dy in gaps:
                if gap < min_gap:
                    min_gap = gap
                    best_dx = dx
                    best_dy = dy

            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                if d < 1e-12:
                    continue
                gap = d - (r[i] + r[j])
                if gap < min_gap:
                    min_gap = gap
                    best_dx = dx / d
                    best_dy = dy / d

            # Try moving in the direction away from the constraint
            for step in [0.002, 0.001, 0.0005, 0.0002]:
                nx = x[i] + best_dx * step
                ny = y[i] + best_dy * step

                if nx < r[i] or nx > w - r[i] or ny < r[i] or ny > h - r[i]:
                    continue

                valid = True
                for j in range(n):
                    if i == j:
                        continue
                    ddx = nx - x[j]
                    ddy = ny - y[j]
                    d2 = ddx * ddx + ddy * ddy
                    rs = r[i] + r[j]
                    if d2 < rs * rs - 1e-9:
                        valid = False
                        break

                if valid:
                    x[i] = nx
                    y[i] = ny
                    break

    return x, y, r


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


def _smooth_objective(z, beta=50.0):
    """
    Smooth log-sum-exp based objective. Uses soft-max to create a differentiable
    surrogate for the max-overlap constraint, enabling better gradient flow.
    """
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # Smooth pairwise overlap: use log-sum-exp of (r_i + r_j - d_ij)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]
    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))

    # overlap_ij = r_i + r_j - d_ij (positive when overlapping)
    overlap_vals = r_sum_pairs - d_pairs

    # Smooth max via log-sum-exp: (1/beta) * log(sum(exp(beta * v)))
    # This gives a smooth approximation to max(overlap_vals, 0)
    shifted = beta * overlap_vals
    shifted = np.clip(shifted, -500, 500)  # numerical safety
    max_shifted = np.max(shifted)
    lse = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    smooth_max_overlap = lse / beta

    # Also penalize individual overlaps with quadratic for local structure
    hard_overlap = np.maximum(overlap_vals, 0.0)
    quad_penalty = 100.0 * np.sum(hard_overlap * hard_overlap)

    obj += 500.0 * smooth_max_overlap + quad_penalty

    # Smooth boundary violations
    left_viol = r - x
    right_viol = x + r - w
    bottom_viol = r - y
    top_viol = y + r - h

    all_viol = np.concatenate([left_viol, right_viol, bottom_viol, top_viol])
    shifted_b = beta * all_viol
    shifted_b = np.clip(shifted_b, -500, 500)
    max_b = np.max(shifted_b)
    lse_b = max_b + np.log(np.sum(np.exp(shifted_b - max_b)))
    smooth_max_viol = lse_b / beta

    hard_viol = np.maximum(all_viol, 0.0)
    obj += 500.0 * smooth_max_viol + 100.0 * np.sum(hard_viol * hard_viol)

    # Aspect ratio penalty (smooth)
    if w < 0.5:
        obj += 200.0 * (0.5 - w) ** 2
    elif w > 1.5:
        obj += 200.0 * (w - 1.5) ** 2

    return obj


def _epigraph_objective(z, beta=30.0):
    """
    Epigraph formulation: z has 3N+2 variables (x, y, r, w, t).
    t is an auxiliary variable representing the maximum constraint violation.
    Objective: -sum(r) + penalty * t + soft constraints (overlap_ij - t <= 0).
    This gives SLSQP a cleaner structure to optimize.
    """
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1 - 1]  # second to last
    t = z[-1]       # last
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r) + 10.0 * max(0.0, t)

    # Pairwise overlap constraints: r_i + r_j - d_ij <= t
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]
    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))

    overlap_vals = r_sum_pairs - d_pairs
    # Violation of epigraph: overlap_vals - t (should be <= 0)
    viol = overlap_vals - t
    # Soft penalty: log-sum-exp of positive violations
    shifted = beta * np.maximum(viol, 0.0)
    shifted = np.clip(shifted, -500, 500)
    max_shifted = np.max(shifted) if shifted.size > 0 else 0.0
    if max_shifted > 0:
        lse = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
        obj += 100.0 * lse / beta
    # Quadratic on hard violations for local structure
    hard_viol = np.maximum(viol, 0.0)
    obj += 50.0 * np.sum(hard_viol * hard_viol)

    # Boundary violations: r_i - x_i <= t, x_i + r_i - w <= t, etc.
    left_viol = r - x - t
    right_viol = x + r - w - t
    bottom_viol = r - y - t
    top_viol = y + r - h - t

    all_bviol = np.concatenate([left_viol, right_viol, bottom_viol, top_viol])
    shifted_b = beta * np.maximum(all_bviol, 0.0)
    shifted_b = np.clip(shifted_b, -500, 500)
    max_b = np.max(shifted_b) if shifted_b.size > 0 else 0.0
    if max_b > 0:
        lse_b = max_b + np.log(np.sum(np.exp(shifted_b - max_b)))
        obj += 100.0 * lse_b / beta
    hard_bviol = np.maximum(all_bviol, 0.0)
    obj += 50.0 * np.sum(hard_bviol * hard_bviol)

    # Aspect ratio penalty
    if w < 0.5:
        obj += 200.0 * (0.5 - w) ** 2
    elif w > 1.5:
        obj += 200.0 * (w - 1.5) ** 2

    return obj


def _simultaneous_pair_move(x, y, r, w, h, i, j, time_budget):
    """
    Try to improve by simultaneously adjusting positions of circles i and j.
    Returns improved (x, y, r) or None.
    """
    x = x.copy()
    y = y.copy()
    n = N
    # Try small relative movements between i and j
    dx = x[i] - x[j]
    dy = y[i] - y[j]
    d = np.sqrt(dx * dx + dy * dy)
    if d < 1e-12:
        return None
    ux = dx / d
    uy = dy / d

    best_sum = np.sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

    for step in [0.005, 0.003, 0.002, 0.001, 0.0005]:
        # Move i away from j, j away from i (spread them)
        nx_i = x[i] + ux * step
        ny_i = y[i] + uy * step
        nx_j = x[j] - ux * step
        ny_j = y[j] - uy * step

        # Check validity
        valid = True
        for k in range(n):
            if k == i or k == j:
                continue
            # Check i against k
            ddx = nx_i - x[k]
            ddy = ny_i - y[k]
            if ddx * ddx + ddy * ddy < (r[i] + r[k]) ** 2 - 1e-9:
                valid = False
                break
            # Check j against k
            ddx = nx_j - x[k]
            ddy = ny_j - y[k]
            if ddx * ddx + ddy * ddy < (r[j] + r[k]) ** 2 - 1e-9:
                valid = False
                break
        if not valid:
            continue

        # Check i and j against each other
        ddx = nx_i - nx_j
        ddy = ny_i - ny_j
        if ddx * ddx + ddy * ddy < (r[i] + r[j]) ** 2 - 1e-9:
            continue

        # Wall checks
        if nx_i < r[i] or nx_i > w - r[i] or ny_i < r[i] or ny_i > h - r[i]:
            continue
        if nx_j < r[j] or nx_j > w - r[j] or ny_j < r[j] or ny_j > h - r[j]:
            continue

        # Try expanding both
        x_trial = x.copy()
        y_trial = y.copy()
        r_trial = r.copy()
        x_trial[i] = nx_i
        y_trial[i] = ny_i
        x_trial[j] = nx_j
        y_trial[j] = ny_j

        # Expand i
        r_max_i = min(nx_i, w - nx_i, ny_i, h - ny_i)
        for k in range(n):
            if k == i:
                continue
            ddx = nx_i - x_trial[k]
            ddy = ny_i - y_trial[k]
            d = np.sqrt(ddx * ddx + ddy * ddy)
            r_max_i = min(r_max_i, d - r_trial[k])
        r_trial[i] = max(0.0, r_max_i)

        # Expand j
        r_max_j = min(nx_j, w - nx_j, ny_j, h - ny_j)
        for k in range(n):
            if k == j:
                continue
            ddx = nx_j - x_trial[k]
            ddy = ny_j - y_trial[k]
            d = np.sqrt(ddx * ddx + ddy * ddy)
            r_max_j = min(r_max_j, d - r_trial[k])
        r_trial[j] = max(0.0, r_max_j)

        if _is_valid(x_trial, y_trial, r_trial, w, h, tol=1e-9):
            s = np.sum(r_trial)
            if s > best_sum + 1e-10:
                best_sum = s
                best_x = x_trial.copy()
                best_y = y_trial.copy()
                best_r = r_trial.copy()

    if best_sum > np.sum(r) + 1e-10:
        return best_x, best_y, best_r
    return None


def _force_relax_unified(z, max_iter=500):
    """Force-directed relaxation in unified parameter space with adaptive step (vectorized)."""
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

        # Wall repulsion (vectorized)
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 100
        fx += right_viol * 100
        fy -= bottom_viol * 100
        fy += top_viol * 100

        # Pairwise repulsion (vectorized)
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

        # Check convergence
        if max_overlap < convergence_tol:
            break

        # Adaptive step size
        force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
        if force_mag > 1.0:
            dt = max(min_dt, dt * 0.5)
        elif force_mag < 0.1:
            dt = min(0.01, dt * 1.1)

        # Update positions
        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        # Clamp positions
        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

    return z


def _greedy_expand(x, y, r, w, h, time_budget):
    """Iteratively expand each circle to its maximum possible radius."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    improved = True
    rounds = 0
    while improved and rounds < 50:
        if time.time() > time_budget:
            break
        improved = False
        rounds += 1
        for i in range(N):
            # Max radius from walls
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            # Max radius from other circles
            for j in range(N):
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
                r[i] = r_max
                improved = True
    return x, y, r


def _local_search_positions(x, y, r, w, h, time_budget):
    """Try small random perturbations of positions (single and multi-circle) to find better packings."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    rng = np.random.default_rng(123)
    best_sum = np.sum(r)
    n = N

    for _ in range(8000):
        if time.time() > time_budget:
            break

        # Decide: single circle (70%) or 2-3 circles (30%)
        if rng.random() < 0.7:
            k = 1
            indices = [rng.integers(0, n)]
        else:
            k = rng.integers(2, 4)
            indices = list(rng.choice(n, size=k, replace=False))

        # Generate perturbations
        perturbations = {}
        for i in indices:
            scale = 0.003 + 0.004 * rng.random()
            perturbations[i] = (rng.normal(0, scale), rng.normal(0, scale))

        # Apply and check validity
        valid = True
        new_x = x.copy()
        new_y = y.copy()
        for i in indices:
            dx, dy = perturbations[i]
            nx = x[i] + dx
            ny = y[i] + dy
            if nx < r[i] or nx > w - r[i] or ny < r[i] or ny > h - r[i]:
                valid = False
                break
            new_x[i] = nx
            new_y[i] = ny

        if not valid:
            continue

        # Check pairwise constraints for all pairs
        for i in range(n):
            for j in range(i + 1, n):
                ddx = new_x[i] - new_x[j]
                ddy = new_y[i] - new_y[j]
                d2 = ddx * ddx + ddy * ddy
                rs = r[i] + r[j]
                if d2 < rs * rs - 1e-9:
                    valid = False
                    break
            if not valid:
                break

        if valid:
            x = new_x
            y = new_y
            # Try to expand all circles
            x2, y2, r2 = _greedy_expand(x, y, r, w, h, time_budget)
            if np.sum(r2) > best_sum:
                best_sum = np.sum(r2)
                r = r2
                x = x2
                y = y2
            else:
                # Revert
                for i in indices:
                    dx, dy = perturbations[i]
                    x[i] -= dx
                    y[i] -= dy
                r = r.copy()

    return x, y, r


def _is_valid_move(x, y, r, w, h, i, new_x, new_y):
    """Check if moving circle i to new_x, new_y is valid."""
    # Wall check
    if new_x < r[i] or new_x > w - r[i]:
        return False
    if new_y < r[i] or new_y > h - r[i]:
        return False

    # Neighbor check
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


def _expansion_relaxation_polish(z, time_budget):
    """
    Advanced polish: Iteratively expand circles and shift positions to create space.
    Improved: multi-step moves, binary search for max radius, more rounds.
    """
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
            # 1. Try to expand radius i to max possible using binary search
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
                # Binary search for the true max valid radius
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

            # 2. If expansion is limited, try to shift circle i to create space
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

            # If touching a constraint, try multi-step moves away
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

                # Try multiple step sizes, from larger to smaller
                for step in [0.05, 0.03, 0.02, 0.01, 0.005, 0.002]:
                    new_x = x[i] + move_x * step
                    new_y = y[i] + move_y * step
                    if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                        x[i] = new_x
                        y[i] = new_y
                        improved = True
                        break
                else:
                    # Try perpendicular directions too
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
                                break
                        if improved:
                            break

        if not improved:
            break

    # Final uniform expansion with binary search
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


def _polish_result(z, time_budget):
    """Polish a valid packing with force-directed position adjustment and radius growth (vectorized)."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dt = 0.001
    max_iter = int(min(3000, time_budget * 150))

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion (vectorized)
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 35
        fx += right_viol * 35
        fy -= bottom_viol * 35
        fy += top_viol * 35

        # Pairwise repulsion (vectorized)
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
            f_pairs = overlap * 18.0
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

        # Adaptive step
        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 0.1:
            dt = 0.0005
        else:
            dt = 0.001

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Try to grow radii uniformly
    grow_factor = 1.0003
    for _ in range(300):
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


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None

    penalty_weight = 1000.0

    def objective(z):
        return _vectorized_objective(z, penalty_weight)

    def smooth_obj(z):
        return _smooth_objective(z, beta=50.0)

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

        # Focused set of aspect ratios (more diverse, including near-optimal ranges)
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 1.2, 0.8,
                         0.9, 1.1, 0.7, 1.4, 0.65, 1.25, 0.78, 1.28, 0.82, 1.15, 0.72, 1.38,
                         0.58, 1.45, 0.62, 1.55, 0.52, 1.65, 0.48, 1.72, 0.45, 1.8, 0.42, 1.85]
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

        # Random starts (more diverse, with varied aspect ratios)
        rng = np.random.default_rng(42)
        for _ in range(25):
            w0 = rng.uniform(0.4, 1.6)
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.04, 0.13)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        # Hexagonal packing initial guess
        w0 = 1.0
        h0 = 1.0
        z = np.zeros(3 * N + 1)
        rows = 5
        cols = 5
        spacing_x = w0 / (cols + 0.5)
        spacing_y = h0 / (rows + 0.5)
        for i in range(N):
            row, col = divmod(i, cols)
            cx = (col + 0.5 + (0.5 if row % 2 == 1 else 0)) * spacing_x
            cy = (row + 0.5) * spacing_y
            r0 = min(spacing_x, spacing_y) * 0.48
            cx = min(max(cx, r0), w0 - r0)
            cy = min(max(cy, r0), h0 - r0)
            z[3 * i] = cx
            z[3 * i + 1] = cy
            z[3 * i + 2] = r0
        z[-1] = w0
        guesses.append(z)

        # Staggered grid with tighter packing
        w0 = 1.0
        h0 = 1.0
        z = np.zeros(3 * N + 1)
        rows = 5
        cols = 5
        sx = w0 / (cols + 0.3)
        sy = h0 / (rows + 0.3)
        for i in range(N):
            row, col = divmod(i, cols)
            cx = (col + 0.5 + (0.5 if row % 2 == 1 else 0)) * sx
            cy = (row + 0.5) * sy
            r0 = min(sx, sy) * 0.49
            cx = min(max(cx, r0), w0 - r0)
            cy = min(max(cy, r0), h0 - r0)
            z[3 * i] = cx
            z[3 * i + 1] = cy
            z[3 * i + 2] = r0
        z[-1] = w0
        guesses.append(z)

        return guesses

    guesses = get_initial_guesses()

    # Phase 1: SLSQP with quadratic objective
    # Prioritize promising aspect ratios (near square to moderately elongated)
    # Sort guesses by expected promise: prefer aspect ratios in [0.7, 1.4]
    def _aspect_promise(z):
        w = z[-1]
        h = 2.0 - w
        # Reward aspect ratios close to golden ratio or known good ranges
        ar = w / h if h > 0.01 else 999
        # Known good: ~0.8-1.3
        if 0.7 <= ar <= 1.4:
            return 0
        elif 0.5 <= ar <= 1.6:
            return 1
        else:
            return 2

    sorted_guesses = sorted(guesses, key=_aspect_promise)
    # Interleave: first pass on best candidates, then remaining
    pass1 = [g for g in sorted_guesses if _aspect_promise(g) == 0]
    pass2 = [g for g in sorted_guesses if _aspect_promise(g) == 1]
    pass3 = [g for g in sorted_guesses if _aspect_promise(g) == 2]
    ordered_guesses = pass1 + pass2 + pass3

    for i, z0 in enumerate(ordered_guesses):
        if time.time() > deadline - 30:
            break

        # Force-directed relaxation to resolve initial overlaps
        z0_relaxed = _force_relax_unified(z0, max_iter=300)

        remaining = deadline - (time.time() - start_time) - 30
        if remaining < 5:
            break
        maxiter = int(min(400, max(100, remaining * 30)))
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
                # Try repair: shrink radii slightly
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.95
                valid, val = check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair.copy()

        except Exception:
            continue

    # Phase 2: SLSQP with smooth objective on top candidates
    if time.time() < deadline - 25:
        # Use the best result so far and several diverse starts
        smooth_starts = []
        if best_result is not None:
            smooth_starts.append(best_result.copy())
        # Add diverse starts from different aspect ratio families
        diverse_indices = [0, 3, 6, 9, 12, 15]  # Spread across sorted guesses
        for idx in diverse_indices:
            if idx < len(ordered_guesses):
                smooth_starts.append(_force_relax_unified(ordered_guesses[idx], max_iter=200))

        for z0 in smooth_starts:
            if time.time() > deadline - 25:
                break
            remaining = deadline - (time.time() - start_time) - 25
            if remaining < 5:
                break
            maxiter = int(min(400, max(100, remaining * 30)))
            try:
                res = minimize(smooth_obj, z0, method='SLSQP',
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

    # Phase 2b: Epigraph formulation with auxiliary variable t
    if time.time() < deadline - 20:
        def epigraph_obj(z_ep):
            return _epigraph_objective(z_ep, beta=30.0)

        # Convert best_result to epigraph space (add t variable)
        epigraph_starts = []
        if best_result is not None:
            z_ep = np.zeros(3 * N + 2)
            z_ep[:3 * N] = best_result[:3 * N]
            z_ep[3 * N] = best_result[-1]  # w
            z_ep[3 * N + 1] = 0.01  # initial t
            epigraph_starts.append(z_ep)
        # Add one more diverse start
        if len(guesses) > 0:
            z0 = _force_relax_unified(guesses[0], max_iter=200)
            z_ep = np.zeros(3 * N + 2)
            z_ep[:3 * N] = z0[:3 * N]
            z_ep[3 * N] = z0[-1]
            z_ep[3 * N + 1] = 0.01
            epigraph_starts.append(z_ep)

        for z0 in epigraph_starts:
            if time.time() > deadline - 20:
                break
            remaining = deadline - (time.time() - start_time) - 20
            if remaining < 5:
                break
            maxiter = int(min(350, max(80, remaining * 25)))
            try:
                # Bounds: x,y in [0,2], r in [0,1], w in [0.3,1.7], t in [0, 0.5]
                bounds_ep = [(0, 2)] * (2 * N) + [(0, 1)] * N + [(0.3, 1.7), (0.0, 0.5)]
                res = minimize(epigraph_obj, z0, method='SLSQP',
                               bounds=bounds_ep,
                               options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 60})

                z_opt = res.x
                # Convert back to standard format
                z_std = z_opt[:3 * N + 1].copy()
                valid, val = check_validity(z_std)

                if valid and val > best_value:
                    best_value = val
                    best_result = z_std.copy()
                elif not valid:
                    z_repair = z_std.copy()
                    r_idx = np.arange(2, len(z_repair) - 1, 3)
                    z_repair[r_idx] *= 0.95
                    valid, val = check_validity(z_repair)
                    if valid and val > best_value:
                        best_value = val
                        best_result = z_repair.copy()

            except Exception:
                continue

    # Phase 3: Polish
    if best_result is not None and time.time() < deadline - 20:
        # 0. Aspect ratio optimization: try small changes to w
        x = best_result[0::3][:N].copy()
        y = best_result[1::3][:N].copy()
        r = best_result[2::3][:N].copy()
        w = best_result[-1]
        h = 2.0 - w

        # Try adjusting aspect ratio
        for dw in [0.02, -0.02, 0.01, -0.01, 0.005, -0.005, 0.002, -0.002]:
            w_trial = w + dw
            h_trial = 2.0 - w_trial
            if w_trial < 0.3 or h_trial < 0.3:
                continue
            # Scale positions and radii to fit new rectangle
            scale_x = w_trial / w
            scale_y = h_trial / h
            x_trial = x * scale_x
            y_trial = y * scale_y
            r_trial = r * min(scale_x, scale_y)
            if _is_valid(x_trial, y_trial, r_trial, w_trial, h_trial, tol=1e-8):
                # Expand
                x2, y2, r2 = _greedy_expand(x_trial, y_trial, r_trial, w_trial, h_trial, deadline - 15)
                if _is_valid(x2, y2, r2, w_trial, h_trial):
                    val = float(np.sum(r2))
                    if val > best_value:
                        best_value = val
                        best_result = np.empty(3 * N + 1)
                        best_result[0::3][:N] = x2
                        best_result[1::3][:N] = y2
                        best_result[2::3][:N] = r2
                        best_result[-1] = w_trial
                        x = x2
                        y = y2
                        r = r2
                        w = w_trial
                        h = h_trial

        # 1. Greedy expansion
        x = best_result[0::3][:N].copy()
        y = best_result[1::3][:N].copy()
        r = best_result[2::3][:N].copy()
        w = best_result[-1]
        h = 2.0 - w

        x, y, r = _greedy_expand(x, y, r, w, h, deadline - 15)
        if _is_valid(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = np.empty(3 * N + 1)
                best_result[0::3][:N] = x
                best_result[1::3][:N] = y
                best_result[2::3][:N] = r
                best_result[-1] = w

        # 2. Force-directed position polish
        if time.time() < deadline - 15:
            polished = _polish_result(best_result, deadline - (time.time() - start_time) - 10)
            if polished is not None:
                valid, val = check_validity(polished)
                if valid and val > best_value:
                    best_value = val
                    best_result = polished

        # 3. Expansion/Relocation Polish
        if best_result is not None and time.time() < deadline - 10:
            expanded = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 5)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # 4. Local search with greedy expansion (increased budget)
        if best_result is not None and time.time() < deadline - 10:
            x = best_result[0::3][:N].copy()
            y = best_result[1::3][:N].copy()
            r = best_result[2::3][:N].copy()
            w = best_result[-1]
            h = 2.0 - w

            x, y, r = _local_search_positions(x, y, r, w, h, deadline - 6)
            if _is_valid(x, y, r, w, h):
                val = float(np.sum(r))
                if val > best_value:
                    best_value = val
                    best_result = np.empty(3 * N + 1)
                    best_result[0::3][:N] = x
                    best_result[1::3][:N] = y
                    best_result[2::3][:N] = r
                    best_result[-1] = w

        # 4b. Simultaneous pair moves
        if best_result is not None and time.time() < deadline - 5:
            x = best_result[0::3][:N].copy()
            y = best_result[1::3][:N].copy()
            r = best_result[2::3][:N].copy()
            w = best_result[-1]
            h = 2.0 - w

            # Try all pairs
            for i in range(N):
                for j in range(i + 1, N):
                    if time.time() > deadline - 4:
                        break
                    result = _simultaneous_pair_move(x, y, r, w, h, i, j, deadline - 4)
                    if result is not None:
                        x_new, y_new, r_new = result
                        if _is_valid(x_new, y_new, r_new, w, h):
                            val = float(np.sum(r_new))
                            if val > best_value:
                                best_value = val
                                x, y, r = x_new, y_new, r_new
                                best_result = np.empty(3 * N + 1)
                                best_result[0::3][:N] = x
                                best_result[1::3][:N] = y
                                best_result[2::3][:N] = r
                                best_result[-1] = w
                if time.time() > deadline - 4:
                    break

        # 5. Constraint-based refinement (new)
        if best_result is not None and time.time() < deadline - 6:
            x = best_result[0::3][:N].copy()
            y = best_result[1::3][:N].copy()
            r = best_result[2::3][:N].copy()
            w = best_result[-1]
            h = 2.0 - w

            x, y, r = _constraint_refine(x, y, r, w, h, deadline - 3)
            if _is_valid(x, y, r, w, h):
                val = float(np.sum(r))
                if val > best_value:
                    best_value = val
                    best_result = np.empty(3 * N + 1)
                    best_result[0::3][:N] = x
                    best_result[1::3][:N] = y
                    best_result[2::3][:N] = r
                    best_result[-1] = w

        # 6. High-precision final polish (new)
        if best_result is not None and time.time() < deadline - 2:
            x = best_result[0::3][:N].copy()
            y = best_result[1::3][:N].copy()
            r = best_result[2::3][:N].copy()
            w = best_result[-1]
            h = 2.0 - w

            x, y, r = _high_precision_polish(x, y, r, w, h, deadline - 1)
            if _is_valid(x, y, r, w, h):
                val = float(np.sum(r))
                if val > best_value:
                    best_value = val
                    best_result = np.empty(3 * N + 1)
                    best_result[0::3][:N] = x
                    best_result[1::3][:N] = y
                    best_result[2::3][:N] = r
                    best_result[-1] = w

        # 7. Final aggressive uniform scaling with binary search (tighter)
        if best_result is not None and time.time() < deadline - 0.5:
            x = best_result[0::3][:N].copy()
            y = best_result[1::3][:N].copy()
            r = best_result[2::3][:N].copy()
            w = best_result[-1]
            h = 2.0 - w

            # Binary search for maximum uniform scale factor (tighter range, more iterations)
            lo, hi = 1.0, 1.02
            for _ in range(60):
                if time.time() > deadline - 0.3:
                    break
                mid = (lo + hi) / 2.0
                r_trial = r * mid
                if _is_valid(x, y, r_trial, w, h, tol=1e-11):
                    lo = mid
                else:
                    hi = mid
            r_final = r * lo
            if _is_valid(x, y, r_final, w, h, tol=1e-10):
                val = float(np.sum(r_final))
                if val > best_value:
                    best_value = val
                    best_result = np.empty(3 * N + 1)
                    best_result[0::3][:N] = x
                    best_result[1::3][:N] = y
                    best_result[2::3][:N] = r_final
                    best_result[-1] = w

        # 8. Final local search + polish cycle (if time remains)
        if best_result is not None and time.time() < deadline - 3:
            x = best_result[0::3][:N].copy()
            y = best_result[1::3][:N].copy()
            r = best_result[2::3][:N].copy()
            w = best_result[-1]
            h = 2.0 - w

            x, y, r = _local_search_positions(x, y, r, w, h, deadline - 2)
            if _is_valid(x, y, r, w, h):
                x, y, r = _greedy_expand(x, y, r, w, h, deadline - 1)
                if _is_valid(x, y, r, w, h):
                    val = float(np.sum(r))
                    if val > best_value:
                        best_value = val
                        best_result = np.empty(3 * N + 1)
                        best_result[0::3][:N] = x
                        best_result[1::3][:N] = y
                        best_result[2::3][:N] = r
                        best_result[-1] = w

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
    for _ in range(50):
        r_trial = r * (1 - 3e-11)
        if _is_valid(x, y, r_trial, w, h, tol=1e-10):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
