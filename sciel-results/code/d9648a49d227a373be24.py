"""Circle packing in perimeter-4 rectangle: SLSQP + Soft-Body Simulation + Polish."""
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


def _soft_body_relax(x, y, r, w, h, time_budget, max_iter=5000):
    """
    Soft-body force-directed relaxation. Circles can temporarily deform
    (reduce effective radius) to squeeze past each other, then rigidify.
    
    Key idea: during relaxation, each circle has an 'effective radius' that
    can be smaller than its target radius when under pressure. This allows
    circles to pass through narrow gaps that would be impossible with rigid
    bodies. The deformation is proportional to local pressure (overlap).
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    n = N

    # Softness parameter: how much circles can deform (0 = rigid, 1 = fully soft)
    # Start soft, gradually rigidify
    softness_schedule = 200  # iterations over which to go from soft to rigid

    dt = 0.003
    for it in range(max_iter):
        if time.time() > time_budget:
            break

        # Softness: 1.0 at start, decays to 0.0
        soft = max(0.0, 1.0 - it / float(softness_schedule))
        # Effective radius = target radius * (1 - softness * deformation)
        # When soft, circles can shrink up to 30% to squeeze through gaps

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 80
        fx += right_viol * 80
        fy -= bottom_viol * 80
        fy += top_viol * 80

        # Pairwise interaction with soft-body behavior
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))

        # Soft-body: effective separation distance can be reduced
        # When soft, we allow some "compression" - the force grows more gradually
        # This simulates elastic deformation
        overlap = r_sum_pairs - d_pairs
        # Soft overlap: when softness > 0, we reduce the effective overlap
        # by allowing circles to "compress" - force is proportional to overlap
        # but with a softer spring constant
        # Hard overlap (rigid): force = overlap * k
        # Soft overlap: force = overlap * k * (1 - softness * min(overlap/r_avg, 1))
        r_avg = 0.5 * r_sum_pairs
        compression_frac = np.minimum(overlap / np.maximum(r_avg, 1e-6), 1.0)
        soft_factor = 1.0 - soft * 0.6 * compression_frac

        f_pairs = overlap * 20.0 * soft_factor

        # Also add a small attractive force when circles are far apart (to pack tightly)
        # This helps the system find dense configurations
        gap = d_pairs - r_sum_pairs
        attraction = np.maximum(gap, 0.0) * 2.0 * (1.0 - soft)

        net_f = f_pairs - attraction

        dx_pairs = dx[iu]
        dy_pairs = dy[iu]
        ux = dx_pairs / np.maximum(d_pairs, 1e-12)
        uy = dy_pairs / np.maximum(d_pairs, 1e-12)

        for idx in range(len(iu[0])):
            i_idx = iu[0][idx]
            j_idx = iu[1][idx]
            f = net_f[idx]
            fx[i_idx] += ux[idx] * f
            fy[i_idx] += uy[idx] * f
            fx[j_idx] -= ux[idx] * f
            fy[j_idx] -= uy[idx] * f

        # Adaptive step size
        force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
        if force_mag > 1.0:
            dt = max(0.0002, dt * 0.7)
        elif force_mag < 0.05:
            dt = min(0.005, dt * 1.05)

        # Update positions
        x += dt * fx
        y += dt * fy

        # Clamp positions to stay within bounds (using target radius)
        x = np.clip(x, r * 0.5, w - r * 0.5)
        y = np.clip(y, r * 0.5, h - r * 0.5)

    # After relaxation, do a greedy expansion to maximize radii
    for _pass in range(20):
        changed = False
        for i in range(n):
            r_max = _max_radius_for_circle(x, y, r, w, h, i)
            if r_max > r[i] + 1e-10:
                r[i] = r_max
                changed = True
        if not changed:
            break

    return x, y, r


def _high_precision_polish(x, y, r, w, h, time_budget):
    """
    High-precision final polish: iterative expansion with very tight tolerances,
    small position adjustments, and uniform scaling.
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    n = N
    tol = 1e-13
    max_rounds = 300

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

        # Try nudging each circle slightly in all 8 directions (finer steps)
        for i in range(n):
            r_max_before = _max_radius_for_circle(x, y, r, w, h, i)
            if r_max_before <= r[i] + tol:
                continue

            for dx in [0.0005, -0.0005, 0.0002, -0.0002, 0.0001, -0.0001, 0.0]:
                for dy in [0.0005, -0.0005, 0.0002, -0.0002, 0.0001, -0.0001, 0.0]:
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
                        # Check if this enables expansion
                        x[i] = nx
                        y[i] = ny
                        r_new_max = _max_radius_for_circle(x, y, r, w, h, i)
                        if r_new_max > r[i] + tol:
                            r[i] = r_new_max
                            improved = True
                            break
                        else:
                            # Keep the position if it's valid (may help other circles)
                            pass

        # Step 3: Uniform scaling attempt with finer granularity
        if not improved:
            for factor in [1.0 + 5e-7, 1.0 + 2e-7, 1.0 + 1e-7, 1.0 + 5e-8, 1.0 + 2e-8, 1.0 + 1e-8]:
                r_trial = r * factor
                if _is_valid(x, y, r_trial, w, h, tol=1e-11):
                    r = r_trial
                    improved = True
                    break

        if not improved:
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
    """Try small random perturbations of positions to find better packings."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    rng = np.random.default_rng(123)
    best_sum = np.sum(r)
    
    for _ in range(5000):
        if time.time() > time_budget:
            break
        # Pick a random circle and nudge it
        i = rng.integers(0, N)
        dx = rng.normal(0, 0.005)
        dy = rng.normal(0, 0.005)
        nx = x[i] + dx
        ny = y[i] + dy
        
        # Check validity of new position
        if nx < r[i] or nx > w - r[i] or ny < r[i] or ny > h - r[i]:
            continue
        valid = True
        for j in range(N):
            if i == j:
                continue
            ddx = nx - x[j]
            ddy = ny - y[j]
            if ddx * ddx + ddy * ddy < (r[i] + r[j]) ** 2 - 1e-9:
                valid = False
                break
        if valid:
            x[i] = nx
            y[i] = ny
            # Try to expand all circles
            x2, y2, r2 = _greedy_expand(x, y, r, w, h, time_budget)
            if np.sum(r2) > best_sum:
                best_sum = np.sum(r2)
                r = r2
                x = x2
                y = y2
            else:
                # Revert
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

    # Phase 0: Soft-body relaxation on top candidates (NEW)
    # This allows circles to squeeze past each other in narrow spaces
    def _aspect_promise(z):
        w = z[-1]
        h = 2.0 - w
        ar = w / h if h > 0.01 else 999
        if 0.7 <= ar <= 1.4:
            return 0
        elif 0.5 <= ar <= 1.6:
            return 1
        else:
            return 2

    sorted_guesses = sorted(guesses, key=_aspect_promise)
    pass1 = [g for g in sorted_guesses if _aspect_promise(g) == 0]
    pass2 = [g for g in sorted_guesses if _aspect_promise(g) == 1]
    pass3 = [g for g in sorted_guesses if _aspect_promise(g) == 2]
    ordered_guesses = pass1 + pass2 + pass3

    # Run soft-body relaxation on the most promising candidates
    soft_body_candidates = ordered_guesses[:12]
    for z0 in soft_body_candidates:
        if time.time() > deadline - 40:
            break
        x0 = z0[0::3][:N].copy()
        y0 = z0[1::3][:N].copy()
        r0 = z0[2::3][:N].copy()
        w0 = z0[-1]
        h0 = 2.0 - w0

        # Soft-body relaxation
        remaining = deadline - (time.time() - start_time) - 40
        if remaining < 3:
            break
        x_sb, y_sb, r_sb = _soft_body_relax(x0, y0, r0, w0, h0, start_time + start_time + remaining * 0.3, max_iter=3000)

        # Check validity and update best
        if _is_valid(x_sb, y_sb, r_sb, w0, h0, tol=1e-8):
            val = float(np.sum(r_sb))
            if val > best_value:
                best_value = val
                best_result = np.empty(3 * N + 1)
                best_result[0::3][:N] = x_sb
                best_result[1::3][:N] = y_sb
                best_result[2::3][:N] = r_sb
                best_result[-1] = w0

    # Phase 1: SLSQP with quadratic objective
    for i, z0 in enumerate(ordered_guesses):
        if time.time() > deadline - 35:
            break

        remaining = deadline - (time.time() - start_time) - 35
        if remaining < 5:
            break
        maxiter = int(min(300, max(80, remaining * 25)))
        try:
            res = minimize(objective, z0, method='SLSQP',
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
        smooth_starts = []
        if best_result is not None:
            smooth_starts.append(best_result.copy())
        diverse_indices = [0, 3, 6, 9, 12, 15]
        for idx in diverse_indices:
            if idx < len(ordered_guesses):
                smooth_starts.append(ordered_guesses[idx].copy())

        for z0 in smooth_starts:
            if time.time() > deadline - 25:
                break
            remaining = deadline - (time.time() - start_time) - 25
            if remaining < 5:
                break
            maxiter = int(min(300, max(80, remaining * 25)))
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

    # Phase 2b: Epigraph formulation
    if time.time() < deadline - 20:
        def epigraph_obj(z_ep):
            return _epigraph_objective(z_ep, beta=30.0)

        epigraph_starts = []
        if best_result is not None:
            z_ep = np.zeros(3 * N + 2)
            z_ep[:3 * N] = best_result[:3 * N]
            z_ep[3 * N] = best_result[-1]
            z_ep[3 * N + 1] = 0.01
            epigraph_starts.append(z_ep)
        if len(guesses) > 0:
            z_ep = np.zeros(3 * N + 2)
            z_ep[:3 * N] = guesses[0][:3 * N]
            z_ep[3 * N] = guesses[0][-1]
            z_ep[3 * N + 1] = 0.01
            epigraph_starts.append(z_ep)

        for z0 in epigraph_starts:
            if time.time() > deadline - 20:
                break
            remaining = deadline - (time.time() - start_time) - 20
            if remaining < 5:
                break
            maxiter = int(min(300, max(80, remaining * 25)))
            try:
                bounds_ep = [(0, 2)] * (2 * N) + [(0, 1)] * N + [(0.3, 1.7), (0.0, 0.5)]
                res = minimize(epigraph_obj, z0, method='SLSQP',
                               bounds=bounds_ep,
                               options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 60})

                z_opt = res.x
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

        # 3. Soft-body relaxation on best result (NEW - second pass)
        if best_result is not None and time.time() < deadline - 12:
            x = best_result[0::3][:N].copy()
            y = best_result[1::3][:N].copy()
            r = best_result[2::3][:N].copy()
            w = best_result[-1]
            h = 2.0 - w
            remaining = deadline - (time.time() - start_time) - 12
            if remaining > 2:
                x_sb, y_sb, r_sb = _soft_body_relax(x, y, r, w, h, start_time + remaining * 0.4, max_iter=2000)
                if _is_valid(x_sb, y_sb, r_sb, w, h, tol=1e-8):
                    val = float(np.sum(r_sb))
                    if val > best_value:
                        best_value = val
                        best_result = np.empty(3 * N + 1)
                        best_result[0::3][:N] = x_sb
                        best_result[1::3][:N] = y_sb
                        best_result[2::3][:N] = r_sb
                        best_result[-1] = w

        # 4. Local search with greedy expansion
        if best_result is not None and time.time() < deadline - 8:
            x = best_result[0::3][:N].copy()
            y = best_result[1::3][:N].copy()
            r = best_result[2::3][:N].copy()
            w = best_result[-1]
            h = 2.0 - w

            x, y, r = _local_search_positions(x, y, r, w, h, deadline - 5)
            if _is_valid(x, y, r, w, h):
                val = float(np.sum(r))
                if val > best_value:
                    best_value = val
                    best_result = np.empty(3 * N + 1)
                    best_result[0::3][:N] = x
                    best_result[1::3][:N] = y
                    best_result[2::3][:N] = r
                    best_result[-1] = w

        # 5. High-precision final polish
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

        # 6. Final aggressive uniform scaling with binary search
        if best_result is not None and time.time() < deadline - 0.5:
            x = best_result[0::3][:N].copy()
            y = best_result[1::3][:N].copy()
            r = best_result[2::3][:N].copy()
            w = best_result[-1]
            h = 2.0 - w

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
    for _ in range(30):
        r_trial = r * (1 - 5e-11)
        if _is_valid(x, y, r_trial, w, h, tol=1e-10):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
