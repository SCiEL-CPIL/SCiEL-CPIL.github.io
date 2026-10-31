"""Circle packing in perimeter-4 rectangle: SLSQP + Advanced Expansion Polish."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

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
    global start_time
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

    # Phase 1: Differential Evolution for global exploration
    if time.time() < deadline - 80:
        de_bounds = [(0, 2)] * (3 * N) + [(0.5, 1.5)]
        de_obj = lambda zv: _vectorized_objective(zv, 1000.0)
        
        try:
            de_res = differential_evolution(
                de_obj, de_bounds, seed=42, maxiter=50, 
                tol=1e-8, popsize=15, mutation=(0.3, 0.8), 
                recombination=0.7, init='latinhypercube'
            )
            if de_res.success or de_res.fun < 0:
                z_de = de_res.x
                valid, val = check_validity(z_de)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_de.copy()
        except Exception:
            pass

    # Phase 2: Local optimization from multiple starts
    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 30:
            break

        # Soft-repulsion relaxation to resolve initial overlaps and create space
        z0_relaxed = _soft_repulsion_relax(z0, max_iter=500, time_budget=deadline - 30)

        # SLSQP optimization
        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(500, max(100, remaining * 40)))
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

    # If DE found something better, use it as the base
    if best_result is None:
        # Fallback: simple grid in unit square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # --- Phase 3: Advanced Polish on best result ---
    if best_result is not None and time.time() < deadline - 25:
        # 1. Force-directed position polish to ensure we are in a good basin
        polished = _polish_result(best_result, deadline - (time.time() - start_time) - 15)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        # 2. Expansion/Relocation Polish
        if best_result is not None and time.time() < deadline - 18:
            expanded = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 10)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # 3. Constraint-based SLSQP refinement
        if best_result is not None and time.time() < deadline - 15:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 8)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # 4. Final quick polish
        if best_result is not None and time.time() < deadline - 5:
            polished2 = _polish_result(best_result, deadline - (time.time() - start_time) - 2)
            if polished2 is not None:
                valid, val = check_validity(polished2)
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

    # Final safety: ensure strict validity with minimal shrink
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))


def _soft_repulsion_relax(z, max_iter=1500, time_budget=None):
    """Soft-repulsion gradient descent: smooth force based on gap (not just overlap) to push circles apart."""
    z = z.copy().astype(float)
    n = N
    iu = np.triu_indices(n, k=1)
    start_t = time.time()

    # Softness parameter: controls how far beyond contact the force reaches
    softness = 0.05  # force extends up to this distance beyond contact
    base_lr = 0.003
    min_lr = 0.0001
    max_lr = 0.008
    lr = base_lr

    for it in range(max_iter):
        if time_budget is not None and time.time() > time_budget:
            break

        x = z[0::3][:n].copy()
        y = z[1::3][:n].copy()
        r = z[2::3][:n].copy()
        w = z[-1]
        h = 2.0 - w

        if w < 0.1 or h < 0.1:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Smooth wall repulsion: quadratic penalty for penetration, linear for proximity
        left_gap = x - r
        right_gap = w - x - r
        bottom_gap = y - r
        top_gap = h - y - r

        # Soft wall force: active within softness distance of wall
        wall_force = np.zeros(n)
        wall_force += np.where(left_gap < softness, (softness - left_gap) / softness, 0.0)
        wall_force_right = np.where(right_gap < softness, (softness - right_gap) / softness, 0.0)
        wall_force_bottom = np.where(bottom_gap < softness, (softness - bottom_gap) / softness, 0.0)
        wall_force_top = np.where(top_gap < softness, (softness - top_gap) / softness, 0.0)

        # Hard wall constraint force (stronger)
        left_pen = np.maximum(-left_gap, 0.0)
        right_pen = np.maximum(-right_gap, 0.0)
        bottom_pen = np.maximum(-bottom_gap, 0.0)
        top_pen = np.maximum(-top_gap, 0.0)

        fx -= wall_force * 10.0 - left_pen * 200.0 + wall_force_right * 10.0 + right_pen * 200.0
        fy -= wall_force_bottom * 10.0 - bottom_pen * 200.0 + wall_force_top * 10.0 + top_pen * 200.0

        # Soft pairwise repulsion: smooth function of gap
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))

        # Gap = distance - sum of radii (negative means overlap)
        gap = d_pairs - r_sum_pairs

        # Soft repulsion: smooth force that is active when gap < softness
        # Force magnitude: smooth bump function centered at gap=0, extending to gap=softness
        # Use a smooth quadratic: f = (softness - gap)^2 / (4 * softness^2) for gap in (-inf, softness)
        # This gives f=1 at gap=-softness, f=0.5 at gap=0, f=0 at gap=softness
        soft_gap = np.clip(softness - gap, 0.0, 2 * softness)
        f_mag = (soft_gap / (2 * softness)) ** 2  # smooth, in [0, 1]

        # Scale: stronger for actual overlap
        overlap_boost = np.where(gap < 0, 5.0, 1.0)
        f_pairs = f_mag * overlap_boost * 15.0

        # Direction
        dx_pairs = dx[iu]
        dy_pairs = dy[iu]
        ux = dx_pairs / d_pairs
        uy = dy_pairs / d_pairs

        # Apply forces
        for idx in range(len(iu[0])):
            i_idx = iu[0][idx]
            j_idx = iu[1][idx]
            fx[i_idx] += ux[idx] * f_pairs[idx]
            fy[i_idx] += uy[idx] * f_pairs[idx]
            fx[j_idx] -= ux[idx] * f_pairs[idx]
            fy[j_idx] -= uy[idx] * f_pairs[idx]

        # Adaptive learning rate
        force_mag = float(np.sqrt(np.sum(fx * fx) + np.sum(fy * fy))) / n
        if force_mag > 1.0:
            lr = max(min_lr, lr * 0.7)
        elif force_mag < 0.05:
            lr = min(max_lr, lr * 1.1)

        # Update positions
        x_new = x + lr * fx
        y_new = y + lr * fy

        # Clamp to valid range
        x_new = np.clip(x_new, r, w - r)
        y_new = np.clip(y_new, r, h - r)

        # Check if we made progress
        x = x_new
        y = y_new
        z[0::3][:n] = x
        z[1::3][:n] = y

        # Convergence check: max overlap should be ~0
        d_sq_final = dx * dx + dy * dy
        d_final = np.sqrt(np.maximum(d_sq_final[iu], 1e-12))
        max_overlap = float(np.max(r_sum_pairs - d_final)) if len(d_final) > 0 else 0.0
        if max_overlap < 1e-8:
            break

    return z


def _polish_result(z, time_budget):
    """Polish a valid packing: soft-repulsion position adjustment + per-circle radius growth."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    iu = np.triu_indices(n, k=1)
    softness = 0.03
    lr = 0.001
    min_lr = 0.00005
    max_lr = 0.004

    max_iter = int(min(8000, time_budget * 400))

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Smooth wall forces
        left_gap = x - r
        right_gap = w - x - r
        bottom_gap = y - r
        top_gap = h - y - r

        left_pen = np.maximum(-left_gap, 0.0)
        right_pen = np.maximum(-right_gap, 0.0)
        bottom_pen = np.maximum(-bottom_gap, 0.0)
        top_pen = np.maximum(-top_gap, 0.0)

        # Soft proximity forces
        lf = np.where(left_gap < softness, (softness - left_gap) / softness, 0.0)
        rf = np.where(right_gap < softness, (softness - right_gap) / softness, 0.0)
        bf = np.where(bottom_gap < softness, (softness - bottom_gap) / softness, 0.0)
        tf = np.where(top_gap < softness, (softness - top_gap) / softness, 0.0)

        fx -= lf * 8.0 - left_pen * 300.0 + rf * 8.0 + right_pen * 300.0
        fy -= bf * 8.0 - bottom_pen * 300.0 + tf * 8.0 + top_pen * 300.0

        # Soft pairwise repulsion
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))

        gap = d_pairs - r_sum_pairs
        soft_gap = np.clip(softness - gap, 0.0, 2 * softness)
        f_mag = (soft_gap / (2 * softness)) ** 2
        overlap_boost = np.where(gap < 0, 8.0, 1.0)
        f_pairs = f_mag * overlap_boost * 12.0

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

        # Adaptive learning rate
        force_mag = float(np.sqrt(np.sum(fx * fx) + np.sum(fy * fy))) / n
        if force_mag > 0.5:
            lr = max(min_lr, lr * 0.7)
        elif force_mag < 0.02:
            lr = min(max_lr, lr * 1.1)

        x += lr * fx
        y += lr * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Per-circle radius expansion with binary search
    for _ in range(30):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            # Max radius for circle i given current positions
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(n):
                if i == j:
                    continue
                dx_j = x[i] - x[j]
                dy_j = y[i] - y[j]
                d_j = np.sqrt(dx_j * dx_j + dy_j * dy_j)
                r_allowed = d_j - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)

            if r_max > r[i] + 1e-7:
                lo = r[i]
                hi = r_max
                for _ in range(20):
                    mid = (lo + hi) / 2.0
                    r_trial = r.copy()
                    r_trial[i] = mid
                    if _is_valid(x, y, r_trial, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-8:
                    r[i] = lo
                    improved = True

        if not improved:
            break

    # Final uniform expansion
    for _ in range(20):
        if time.time() > time_budget:
            break
        lo, hi = 1.0, 1.005
        for _ in range(12):
            mid = (lo + hi) / 2.0
            r_trial = r * mid
            if _is_valid(x, y, r_trial, w, h):
                lo = mid
            else:
                hi = mid
        if lo > 1.0 + 1e-10:
            r = r * lo
        else:
            break

    z_new = np.empty(3 * N + 1)
    z_new[0::3][:n] = x
    z_new[1::3][:n] = y
    z_new[2::3][:n] = r
    z_new[-1] = w

    if _is_valid(x, y, r, w, h):
        return z_new
    return None


def _expansion_relaxation_polish(z, time_budget):
    """
    Advanced polish: soft-repulsion repositioning + iterative per-circle radius expansion.
    Uses smooth gap-based forces to find better positions, then greedily expands radii.
    """
    global start_time
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    iu = np.triu_indices(n, k=1)
    softness = 0.04
    lr = 0.002
    min_lr = 0.00005
    max_lr = 0.005

    # Phase 1: Soft-repulsion position optimization (multiple rounds)
    max_pos_rounds = 3
    for pos_rnd in range(max_pos_rounds):
        if time.time() > time_budget:
            break

        max_iter = int(min(3000, (time_budget - (time.time() - start_time)) * 300))
        for it in range(max_iter):
            if time.time() > time_budget:
                break

            fx = np.zeros(n)
            fy = np.zeros(n)

            # Wall forces
            left_gap = x - r
            right_gap = w - x - r
            bottom_gap = y - r
            top_gap = h - y - r

            left_pen = np.maximum(-left_gap, 0.0)
            right_pen = np.maximum(-right_gap, 0.0)
            bottom_pen = np.maximum(-bottom_gap, 0.0)
            top_pen = np.maximum(-top_gap, 0.0)

            lf = np.where(left_gap < softness, (softness - left_gap) / softness, 0.0)
            rf = np.where(right_gap < softness, (softness - right_gap) / softness, 0.0)
            bf = np.where(bottom_gap < softness, (softness - bottom_gap) / softness, 0.0)
            tf = np.where(top_gap < softness, (softness - top_gap) / softness, 0.0)

            fx -= lf * 10.0 - left_pen * 300.0 + rf * 10.0 + right_pen * 300.0
            fy -= bf * 10.0 - bottom_pen * 300.0 + tf * 10.0 + top_pen * 300.0

            # Pairwise soft repulsion
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d_sq = dx * dx + dy * dy
            r_sum = r[:, None] + r[None, :]

            d_sq_pairs = d_sq[iu]
            r_sum_pairs = r_sum[iu]
            d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))

            gap = d_pairs - r_sum_pairs
            soft_gap = np.clip(softness - gap, 0.0, 2 * softness)
            f_mag = (soft_gap / (2 * softness)) ** 2
            overlap_boost = np.where(gap < 0, 10.0, 1.0)
            f_pairs = f_mag * overlap_boost * 15.0

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

            force_mag = float(np.sqrt(np.sum(fx * fx) + np.sum(fy * fy))) / n
            if force_mag > 0.5:
                lr = max(min_lr, lr * 0.7)
            elif force_mag < 0.02:
                lr = min(max_lr, lr * 1.1)

            x += lr * fx
            y += lr * fy
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)

    # Phase 2: Iterative per-circle radius expansion
    max_expand_rounds = 40
    for exp_rnd in range(max_expand_rounds):
        if time.time() > time_budget:
            break

        improved = False
        for i in range(n):
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(n):
                if i == j:
                    continue
                dx_j = x[i] - x[j]
                dy_j = y[i] - y[j]
                d_j = np.sqrt(dx_j * dx_j + dy_j * dy_j)
                r_allowed = d_j - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)

            if r_max > r[i] + 1e-7:
                lo = r[i]
                hi = r_max
                for _ in range(22):
                    mid = (lo + hi) / 2.0
                    r_trial = r.copy()
                    r_trial[i] = mid
                    if _is_valid(x, y, r_trial, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-8:
                    r[i] = lo
                    improved = True

        if not improved:
            break

    # Phase 3: Final uniform expansion
    for _ in range(30):
        if time.time() > time_budget:
            break
        lo, hi = 1.0, 1.01
        for _ in range(15):
            mid = (lo + hi) / 2.0
            r_trial = r * mid
            if _is_valid(x, y, r_trial, w, h):
                lo = mid
            else:
                hi = mid
        if lo > 1.0 + 1e-10:
            r = r * lo
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


def _max_radius_for_circle(x, y, r, w, h, i):
    """Calculate maximum possible radius for circle i given other positions."""
    r_wall = min(x[i], w - x[i], y[i], h - y[i])
    r_max = r_wall
    for j in range(N):
        if i == j: continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx*dx + dy*dy)
        r_allowed = d - r[j]
        if r_allowed < r_max:
            r_max = r_allowed
    return max(0.0, r_max)


def _is_valid_move(x, y, r, w, h, i, new_x, new_y):
    """Check if moving circle i to new_x, new_y is valid."""
    # Wall check
    if new_x < r[i] or new_x > w - r[i]:
        return False
    if new_y < r[i] or new_y > h - r[i]:
        return False
    
    # Neighbor check
    for j in range(N):
        if i == j: continue
        dx = new_x - x[j]
        dy = new_y - y[j]
        d2 = dx*dx + dy*dy
        r_sum = r[i] + r[j]
        if d2 < r_sum*r_sum - 1e-9:
            return False
    return True


def _constraint_refine(z, time_budget):
    """High-precision SLSQP refinement using explicit constraints with tighter tolerances."""
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

    def make_boundary_constraint(idx, axis, side):
        if axis == 0:
            if side == 0:
                return lambda p: p[3 * idx] - p[3 * idx + 2]
            else:
                return lambda p: p[-1] - p[3 * idx] - p[3 * idx + 2]
        else:
            if side == 0:
                return lambda p: p[3 * idx + 1] - p[3 * idx + 2]
            else:
                return lambda p: (2.0 - p[-1]) - p[3 * idx + 1] - p[3 * idx + 2]

    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 0, 0)})
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 0, 1)})
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 1, 0)})
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 1, 1)})

    for i in range(n):
        for j in range(i + 1, n):
            def make_overlap_constraint(idx1, idx2):
                return lambda p: (
                    (p[3 * idx1] - p[3 * idx2]) ** 2 +
                    (p[3 * idx1 + 1] - p[3 * idx2 + 1]) ** 2 -
                    (p[3 * idx1 + 2] + p[3 * idx2 + 2]) ** 2
                )
            constraints.append({'type': 'ineq', 'fun': make_overlap_constraint(i, j)})

    maxiter = int(min(1200, time_budget * 200))
    try:
        res = minimize(obj_refine, z, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-15, 'maxls': 150})

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
