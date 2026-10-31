"""Circle packing in perimeter-4 rectangle: SLSQP + Advanced Expansion Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _soft_max(vals, temp=1.0):
    """Smooth approximation of max using log-sum-exp."""
    vals = np.asarray(vals)
    if vals.size == 0:
        return 0.0
    max_val = np.max(vals)
    if max_val < -100 * temp:
        return max_val
    exp_vals = np.exp((vals - max_val) / temp)
    return max_val + temp * np.log(np.sum(exp_vals))


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


def _dual_primal_inner(z_fixed_w, time_budget, temp=0.005):
    """
    Inner solve: fix aspect ratio w, optimize circle positions and radii.
    Uses smooth epigraph formulation: minimize -sum(r) + penalty * soft_max(violations)
    where violations are boundary and pairwise overlap amounts.
    """
    n = N
    w = z_fixed_w[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2:
        return None

    iu = np.triu_indices(n, k=1)
    iu0 = iu[0]
    iu1 = iu[1]

    # Extract initial state (positions and radii only, w is fixed)
    x0 = z_fixed_w[0::3][:n].copy()
    y0 = z_fixed_w[1::3][:n].copy()
    r0 = z_fixed_w[2::3][:n].copy()

    def obj_inner(v):
        """v = [x(21), y(21), r(21)] - w is fixed externally"""
        x = v[0:n]
        y = v[n:2*n]
        r = v[2*n:3*n]

        if np.any(r < 0.001):
            return 1e6

        # Boundary violations (positive = violation)
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)

        # Pairwise violations (positive = overlap)
        dx = x[iu0] - x[iu1]
        dy = y[iu0] - y[iu1]
        d_sq = dx * dx + dy * dy
        r_sum = r[iu0] + r[iu1]
        d_pairs = np.sqrt(np.maximum(d_sq, 1e-12))
        pair_viol = np.maximum(r_sum - d_pairs, 0.0)

        all_viol = np.concatenate([left_viol, right_viol, bottom_viol, top_viol, pair_viol])
        sm = _soft_max(all_viol, temp)

        return -np.sum(r) + 800.0 * sm

    v0 = np.concatenate([x0, y0, r0])

    # Bounds for inner variables
    bounds = [(0, 2)] * (2 * n) + [(0.001, 1.0)] * n

    maxiter = int(min(3000, time_budget * 500))
    try:
        res = minimize(obj_inner, v0, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 200, 'disp': False})

        v_opt = res.x
        x_opt = v_opt[0:n]
        y_opt = v_opt[n:2*n]
        r_opt = v_opt[2*n:3*n]

        # Validate
        if not _is_valid(x_opt, y_opt, r_opt, w, h):
            # Try slight shrink
            r_shrunk = r_opt * 0.995
            if _is_valid(x_opt, y_opt, r_shrunk, w, h):
                r_opt = r_shrunk
            else:
                return None

        # Reconstruct z with fixed w
        z_new = np.empty(3 * n + 1)
        z_new[0::3][:n] = x_opt
        z_new[1::3][:n] = y_opt
        z_new[2::3][:n] = r_opt
        z_new[-1] = w
        return z_new

    except Exception:
        return None


def _dual_primal_alternating(z_init, time_budget):
    """
    Dual-primal alternating scheme:
    1. Fix aspect ratio w, solve inner placement (maximize sum of radii)
    2. Compute subgradient of dual w.r.t. w
    3. Update w using subgradient
    4. Repeat until convergence or time budget
    """
    n = N
    z = z_init.copy().astype(float)
    best_z = z.copy()
    best_val = 0.0

    # Initial validity check
    x = z[0::3][:n]
    y = z[1::3][:n]
    r = z[2::3][:n]
    w = z[-1]
    h = 2.0 - w
    if _is_valid(x, y, r, w, h):
        best_val = float(np.sum(r))

    # Subgradient step size schedule
    base_lr = 0.05
    lr_decay = 0.92
    lr = base_lr

    iu = np.triu_indices(n, k=1)
    iu0 = iu[0]
    iu1 = iu[1]

    max_rounds = int(min(40, time_budget * 2))

    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        w_curr = z[-1]
        h_curr = 2.0 - w_curr

        # --- Inner solve: fix w, optimize placement ---
        inner_budget = min(8.0, (time_budget - (time.time() - time_budget)) * 0.6)
        inner_budget = min(inner_budget, time_budget - time.time())
        if inner_budget < 2.0:
            break

        z_inner = _dual_primal_inner(z, inner_budget)

        if z_inner is not None:
            x_i = z_inner[0::3][:n]
            y_i = z_inner[1::3][:n]
            r_i = z_inner[2::3][:n]
            val_i = float(np.sum(r_i))

            if val_i > best_val:
                best_val = val_i
                best_z = z_inner.copy()

            z = z_inner

        # --- Compute subgradient of dual w.r.t. w ---
        # The dual function g(w) = max_{placement} sum(r) s.t. constraints(w)
        # Subgradient: d/dw of the objective contribution from boundary constraints
        # When a circle touches the left/right wall, increasing w helps (positive subgrad)
        # When a circle touches top/bottom wall, increasing w hurts (negative subgrad via h=2-w)

        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w

        # Boundary contact indicators
        left_touch = np.abs(x - r) < 0.02
        right_touch = np.abs(x + r - w) < 0.02
        bottom_touch = np.abs(y - r) < 0.02
        top_touch = np.abs(y + r - h) < 0.02

        # Subgradient: 
        # - Left/right walls: increasing w relaxes right constraint, tightens left (if x-r=0)
        #   Net effect on sum(r): circles at right wall benefit from larger w
        # - Top/bottom walls: h = 2-w, so increasing w decreases h, tightening top/bottom
        #   Circles at top/bottom are hurt by increasing w
        subgrad_w = (np.sum(left_touch) + np.sum(right_touch)) - (np.sum(bottom_touch) + np.sum(top_touch))

        # Also consider: if all circles are interior, the subgradient is 0 (no active constraints)
        # Scale by number of circles for stability
        subgrad_w = subgrad_w / max(1, n)

        # --- Update w using subgradient ---
        # Projected subgradient: w_new = w + lr * subgrad_w, projected to [0.3, 1.7]
        w_new = w + lr * subgrad_w
        w_new = np.clip(w_new, 0.3, 1.7)

        # Only update if it's a meaningful change
        if abs(w_new - w) > 1e-4:
            z[-1] = w_new

        # Decay learning rate
        lr *= lr_decay

        # Early stop if subgradient is tiny
        if abs(subgrad_w) < 0.01:
            break

    return best_z


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

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

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 35:
            break

        # Force-directed relaxation to resolve initial overlaps
        z0_relaxed = _force_relax_unified(z0, max_iter=500)

        # SLSQP optimization
        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(600, max(150, remaining * 50)))
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

    # --- Phase 2: Dual-Primal Alternating Scheme ---
    if best_result is not None and time.time() < deadline - 30:
        dp_budget = deadline - (time.time() - start_time) - 25
        dp_result = _dual_primal_alternating(best_result, dp_budget)
        if dp_result is not None:
            valid, val = check_validity(dp_result)
            if valid and val > best_value:
                best_value = val
                best_result = dp_result

    # --- Phase 3: Advanced Polish on best result ---
    if best_result is not None and time.time() < deadline - 25:
        # 1. Force-directed position polish to ensure we are in a good basin
        polished = _polish_result(best_result, deadline - (time.time() - start_time) - 18)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        # 2. Expansion/Relocation Polish (16-direction adaptive)
        if best_result is not None and time.time() < deadline - 18:
            expanded = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 10)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # 3. Second dual-primal pass after expansion (exploit new basin)
        if best_result is not None and time.time() < deadline - 15:
            dp_budget2 = deadline - (time.time() - start_time) - 10
            dp_result2 = _dual_primal_alternating(best_result, dp_budget2)
            if dp_result2 is not None:
                valid, val = check_validity(dp_result2)
                if valid and val > best_value:
                    best_value = val
                    best_result = dp_result2

        # 4. Constraint-based SLSQP refinement with tight tolerance
        if best_result is not None and time.time() < deadline - 12:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 7)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # 5. Post-SLSQP per-circle expansion
        if best_result is not None and time.time() < deadline - 8:
            expanded2 = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 4)
            if expanded2 is not None:
                valid, val = check_validity(expanded2)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded2

        # 6. Final quick polish
        if best_result is not None and time.time() < deadline - 4:
            polished2 = _polish_result(best_result, deadline - (time.time() - start_time) - 1)
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
    for _ in range(20):
        r_trial = r * (1 - 1e-10)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    # One final micro-expansion pass
    for _ in range(30):
        lo, hi = 1.0, 1.005
        for _ in range(12):
            mid = (lo + hi) / 2.0
            r_trial = r * mid
            if _is_valid(x, y, r_trial, w, h):
                lo = mid
            else:
                hi = mid
        if lo > 1.0 + 1e-12:
            r = r * lo
        else:
            break

    return np.column_stack((x, y, r))


def _force_relax_unified(z, max_iter=800):
    """Force-directed relaxation with fully vectorized pair force accumulation."""
    z = z.copy().astype(float)
    n = N
    dt = 0.01
    min_dt = 0.0001
    convergence_tol = 1e-9
    iu = np.triu_indices(n, k=1)
    iu0 = iu[0]
    iu1 = iu[1]

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
        fx -= left_viol * 150
        fx += right_viol * 150
        fy -= bottom_viol * 150
        fy += top_viol * 150

        # Pairwise repulsion (vectorized with np.add.at)
        dx = x[iu0] - x[iu1]
        dy = y[iu0] - y[iu1]
        d_sq = dx * dx + dy * dy
        r_sum = r[iu0] + r[iu1]
        d_pairs = np.sqrt(np.maximum(d_sq, 1e-12))
        overlap = np.maximum(r_sum - d_pairs, 0.0)
        max_overlap = float(np.max(overlap)) if overlap.size > 0 else 0.0

        if max_overlap > convergence_tol:
            f_pairs = overlap * 20.0
            ux = dx / d_pairs
            uy = dy / d_pairs
            fx_i = ux * f_pairs
            fy_i = uy * f_pairs
            np.add.at(fx, iu0, fx_i)
            np.add.at(fy, iu0, fy_i)
            np.add.at(fx, iu1, -fx_i)
            np.add.at(fy, iu1, -fy_i)

        # Check convergence
        if max_overlap < convergence_tol:
            break

        # Adaptive step size
        force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
        if force_mag > 1.0:
            dt = max(min_dt, dt * 0.5)
        elif force_mag < 0.1:
            dt = min(0.02, dt * 1.15)

        # Update positions
        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        # Clamp positions (vectorized)
        z[0::3][:n] = np.clip(z[0::3][:n], r, w - r)
        z[1::3][:n] = np.clip(z[1::3][:n], r, h - r)

    return z


def _polish_result(z, time_budget):
    """Polish a valid packing with vectorized force-directed adjustment and radius growth."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dt = 0.003
    max_iter = int(min(8000, time_budget * 400))
    iu = np.triu_indices(n, k=1)
    iu0 = iu[0]
    iu1 = iu[1]

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
        fx -= left_viol * 50
        fx += right_viol * 50
        fy -= bottom_viol * 50
        fy += top_viol * 50

        # Pairwise repulsion (vectorized with np.add.at)
        dx = x[iu0] - x[iu1]
        dy = y[iu0] - y[iu1]
        d_sq = dx * dx + dy * dy
        r_sum = r[iu0] + r[iu1]
        d_pairs = np.sqrt(np.maximum(d_sq, 1e-12))
        overlap = np.maximum(r_sum - d_pairs, 0.0)

        if np.max(overlap) > 1e-10:
            f_pairs = overlap * 25.0
            ux = dx / d_pairs
            uy = dy / d_pairs
            fx_i = ux * f_pairs
            fy_i = uy * f_pairs
            np.add.at(fx, iu0, fx_i)
            np.add.at(fy, iu0, fy_i)
            np.add.at(fx, iu1, -fx_i)
            np.add.at(fy, iu1, -fy_i)

        # Adaptive step
        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 0.5:
            dt = 0.001
        elif max_force > 0.1:
            dt = 0.002
        else:
            dt = 0.004

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Try to grow radii uniformly with binary search
    for _ in range(60):
        if time.time() > time_budget:
            break
        lo, hi = 1.0, 1.02
        for _ in range(16):
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

    # Reconstruct z
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
    Advanced polish: Systematic multi-directional position search + per-circle radius expansion.
    Uses 16-direction local search with geometric step decay for finer exploration.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # 16 directions for local search (finer angular resolution)
    n_dirs = 16
    angles = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
    dirs_x = np.cos(angles)
    dirs_y = np.sin(angles)
    # Geometric step decay: start large, shrink quickly
    base_steps = [0.1, 0.06, 0.035, 0.02, 0.012, 0.007, 0.004, 0.002, 0.001]

    max_rounds = 80
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        improved = False
        for i in range(n):
            # Try to reposition circle i in 16 directions to create more space
            best_pos = None
            best_gap = -1.0
            for di in range(n_dirs):
                for step in base_steps:
                    nx = x[i] + dirs_x[di] * step
                    ny = y[i] + dirs_y[di] * step
                    if nx < r[i] or nx > w - r[i]:
                        continue
                    if ny < r[i] or ny > h - r[i]:
                        continue
                    # Check validity and compute minimum gap to neighbors
                    valid = True
                    min_gap = float('inf')
                    for j in range(n):
                        if i == j:
                            continue
                        dx = nx - x[j]
                        dy = ny - y[j]
                        d = np.sqrt(dx * dx + dy * dy)
                        gap = d - (r[i] + r[j])
                        if gap < 1e-9:
                            valid = False
                            break
                        if gap < min_gap:
                            min_gap = gap
                    if valid and min_gap > best_gap:
                        best_gap = min_gap
                        best_pos = (nx, ny)
            
            if best_pos is not None and best_gap > 0.006:
                x[i] = best_pos[0]
                y[i] = best_pos[1]
                improved = True

            # Expand radius i to maximum possible
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
                lo = r[i]
                hi = r_max
                for _ in range(24):
                    mid = (lo + hi) / 2.0
                    r_trial = r.copy()
                    r_trial[i] = mid
                    if _is_valid(x, y, r_trial, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-7:
                    r[i] = lo
                    improved = True

        if not improved:
            break

    # Per-circle radius expansion with positional micro-adjustment
    for pass_num in range(3):
        if time.time() > time_budget:
            break
        any_improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            # Try expanding circle i
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

            if r_max > r[i] + 1e-7:
                # Binary search for max valid radius
                lo_r = r[i]
                hi_r = r_max
                for _ in range(20):
                    mid_r = (lo_r + hi_r) / 2.0
                    r_trial = r.copy()
                    r_trial[i] = mid_r
                    if _is_valid(x, y, r_trial, w, h):
                        lo_r = mid_r
                    else:
                        hi_r = mid_r
                if lo_r > r[i] + 1e-8:
                    r[i] = lo_r
                    any_improved = True

        if not any_improved:
            break

    # Final uniform expansion with binary search
    for _ in range(80):
        if time.time() > time_budget:
            break
        lo, hi = 1.0, 1.04
        for _ in range(20):
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
    """High-precision SLSQP refinement using constraint-based formulation with tight tolerance."""
    z = z.copy().astype(float)
    n = N
    iu = np.triu_indices(n, k=1)
    iu0 = iu[0]
    iu1 = iu[1]

    def obj_refine(zv):
        r = zv[2::3][:n]
        return -np.sum(r)

    def boundary_constraints(zv):
        """Return vector of all boundary constraint values (must be >= 0)."""
        x = zv[0::3][:n]
        y = zv[1::3][:n]
        r = zv[2::3][:n]
        w = zv[-1]
        h = 2.0 - w
        left = x - r
        right = w - x - r
        bottom = y - r
        top = h - y - r
        return np.concatenate([left, right, bottom, top])

    def pairwise_constraints(zv):
        """Return vector of all pairwise distance constraint values (must be >= 0)."""
        x = zv[0::3][:n]
        y = zv[1::3][:n]
        r = zv[2::3][:n]
        dx = x[iu0] - x[iu1]
        dy = y[iu0] - y[iu1]
        d2 = dx * dx + dy * dy
        r_sum = r[iu0] + r[iu1]
        return d2 - r_sum * r_sum

    constraints = [
        {'type': 'ineq', 'fun': boundary_constraints},
        {'type': 'ineq', 'fun': pairwise_constraints}
    ]

    bounds = [(0, 2), (0, 2), (0.001, 1.0)] * n + [(0.3, 1.7)]

    maxiter = int(min(2500, time_budget * 400))
    try:
        res = minimize(obj_refine, z, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-16, 'maxls': 300, 'disp': False})

        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w

        if _is_valid(x, y, r, w, h):
            return z_opt
        # Try repair with slight shrink
        z_repair = z_opt.copy()
        r_idx = np.arange(2, len(z_repair) - 1, 3)
        z_repair[r_idx] *= 0.995
        x = z_repair[0::3][:n]
        y = z_repair[1::3][:n]
        r = z_repair[2::3][:n]
        w = z_repair[-1]
        h = 2.0 - w
        if _is_valid(x, y, r, w, h):
            return z_repair
        return None
    except Exception:
        return None


def _soft_constraint_refine(z, time_budget):
    """SLSQP refinement using soft-max epigraph: minimize -sum(r) + penalty*soft_max(violations)."""
    z = z.copy().astype(float)
    n = N
    iu = np.triu_indices(n, k=1)
    iu0 = iu[0]
    iu1 = iu[1]
    temp = 0.01  # temperature for soft-max (smaller = sharper)

    def obj_soft(zv):
        x = zv[0::3][:n]
        y = zv[1::3][:n]
        r = zv[2::3][:n]
        w = zv[-1]
        h = 2.0 - w
        if w < 0.1 or h < 0.1:
            return 1e6
        # Boundary violations (negative = violation)
        left = x - r
        right = w - x - r
        bottom = y - r
        top = h - y - r
        # Pairwise violations
        dx = x[iu0] - x[iu1]
        dy = y[iu0] - y[iu1]
        d2 = dx * dx + dy * dy
        r_sum = r[iu0] + r[iu1]
        pair_viol = r_sum * r_sum - d2  # positive = violation

        all_viol = np.concatenate([-left, -right, -bottom, -top, pair_viol])
        # soft-max of violations (penalize the worst violation smoothly)
        sm = _soft_max(all_viol, temp)
        penalty = 500.0
        return -np.sum(r) + penalty * sm

    bounds = [(0, 2), (0, 2), (0.001, 1.0)] * n + [(0.3, 1.7)]
    maxiter = int(min(2000, time_budget * 350))
    try:
        res = minimize(obj_soft, z, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 200, 'disp': False})
        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w
        if _is_valid(x, y, r, w, h):
            return z_opt
        # Try repair
        z_repair = z_opt.copy()
        r_idx = np.arange(2, len(z_repair) - 1, 3)
        z_repair[r_idx] *= 0.998
        x = z_repair[0::3][:n]
        y = z_repair[1::3][:n]
        r = z_repair[2::3][:n]
        w = z_repair[-1]
        h = 2.0 - w
        if _is_valid(x, y, r, w, h):
            return z_repair
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
