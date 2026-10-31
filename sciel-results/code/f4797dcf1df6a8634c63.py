"""Circle packing in perimeter-4 rectangle: SLSQP + ADMM + Advanced Expansion Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _admm_solve(z0, time_budget, rho=100.0):
    """
    ADMM solver for circle packing using proximal decomposition.
    
    Formulation:
    min  -sum(r_i) + (rho/2) * sum_pairs [max(0, r_i+r_j - d_ij)^2] + (rho/2) * sum_walls [violation^2]
    s.t.  x_i in [r_i, w-r_i], y_i in [r_i, h-r_i], r_i > 0
    
    We use a simplified ADMM where:
    - x-update: project onto feasible region considering current r
    - r-update: maximize sum(r) subject to proximity to previous r and constraint penalties
    """
    n = N
    x = z0[0::3][:n].copy().astype(float)
    y = z0[1::3][:n].copy().astype(float)
    r = z0[2::3][:n].copy().astype(float)
    w = float(z0[-1])
    h = 2.0 - w

    n_pairs = n * (n - 1) // 2
    iu = np.triu_indices(n, k=1)
    
    # Multipliers
    lam_cc = np.zeros(n_pairs)
    lam_left = np.zeros(n)
    lam_right = np.zeros(n)
    lam_bottom = np.zeros(n)
    lam_top = np.zeros(n)

    rho_cc = rho
    rho_wall = rho
    max_iter = int(min(3000, time_budget * 150))
    
    best_val = float(np.sum(r)) if _is_valid(x, y, r, w, h) else -1e10
    best_state = (x.copy(), y.copy(), r.copy(), w) if best_val > -1e9 else None

    # Adaptive rho
    rho_factor = 1.0

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        # --- X-update: Move circles to reduce violations ---
        # For each circle, compute the optimal position that balances
        # proximity to current position and constraint satisfaction
        
        # Build force field
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Circle-circle repulsion with augmented Lagrangian
        dx_mat = x[:, None] - x[None, :]
        dy_mat = y[:, None] - y[None, :]
        d_sq_mat = dx_mat * dx_mat + dy_mat * dy_mat
        d_pairs = np.sqrt(np.maximum(d_sq_mat[iu], 1e-12))
        r_sum_pairs = r[iu[0]] + r[iu[1]]
        
        # g_ij = d_ij - (r_i + r_j), want g_ij >= 0
        # Augmented penalty: lam_ij * g_ij + (rho/2) * g_ij^2
        # Gradient wrt x_i: -lam_ij * (dx_ij/d_ij) - rho * g_ij * (dx_ij/d_ij)
        # = -(lam_ij + rho * g_ij) * (dx_ij/d_ij)
        # But we want to minimize, so move in direction that increases g_ij when violated
        
        g_cc = d_pairs - r_sum_pairs  # positive means feasible
        # Force magnitude: if g_cc < 0 (overlap), push apart
        # Effective: (lam + rho * min(0, g)) gives force only when violated
        # Better: use (lam + rho * g) but clip to push apart only when needed
        force_cc = lam_cc + rho_cc * np.minimum(g_cc, 0.0)
        # Also add base repulsion for stability
        force_cc += rho_cc * 0.1 * np.maximum(-g_cc, 0.0)
        
        dx_pairs = dx_mat[iu]
        dy_pairs = dy_mat[iu]
        ux = dx_pairs / np.maximum(d_pairs, 1e-12)
        uy = dy_pairs / np.maximum(d_pairs, 1e-12)
        
        for idx in range(len(iu[0])):
            i_idx = iu[0][idx]
            j_idx = iu[1][idx]
            f = force_cc[idx]
            fx[i_idx] += ux[idx] * f
            fy[i_idx] += uy[idx] * f
            fx[j_idx] -= ux[idx] * f
            fy[j_idx] -= uy[idx] * f
        
        # Wall forces
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        
        fx -= (lam_left + rho_wall * left_viol)
        fx += (lam_right + rho_wall * right_viol)
        fy -= (lam_bottom + rho_wall * bottom_viol)
        fy += (lam_top + rho_wall * top_viol)
        
        # Step
        step_size = 0.05 / (1.0 + np.max(np.abs(fx)) / 10.0 + np.max(np.abs(fy)) / 10.0)
        x += step_size * fx
        y += step_size * fy
        
        # Project onto feasible region (walls)
        for i in range(n):
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])
        
        # --- R-update: Maximize sum(r) with constraints ---
        # For each circle, the max radius is limited by:
        # 1. Walls: min(x_i, w-x_i, y_i, h-y_i)
        # 2. Other circles: d_ij - r_j for all j != i
        # 3. Proximity to previous r (ADMM regularization)
        
        for i in range(n):
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
            r_max = max(0.001, r_max)
            
            # ADMM: balance between r_max and previous r + multiplier
            # r_new = max(0, min(r_max, r_old + (lam_related) / rho))
            # Simplified: move toward r_max but with regularization
            r_target = r_max
            r_reg = r[i]
            # Blend
            alpha = 0.5  # how aggressively to grow
            r[i] = np.clip(r[i] + alpha * (r_target - r[i]), 0.001, r_max)
        
        # --- Multiplier updates ---
        dx_mat = x[:, None] - x[None, :]
        dy_mat = y[:, None] - y[None, :]
        d_sq_mat = dx_mat * dx_mat + dy_mat * dy_mat
        d_pairs = np.sqrt(np.maximum(d_sq_mat[iu], 1e-12))
        r_sum_pairs = r[iu[0]] + r[iu[1]]
        g_cc = d_pairs - r_sum_pairs
        lam_cc += rho_cc * g_cc
        lam_cc = np.clip(lam_cc, -50 * rho_factor, 50 * rho_factor)
        
        left_viol = r - x
        right_viol = w - x - r
        bottom_viol = r - y
        top_viol = h - y - r
        
        lam_left += rho_wall * left_viol
        lam_right += rho_wall * right_viol
        lam_bottom += rho_wall * bottom_viol
        lam_top += rho_wall * top_viol
        
        lam_left = np.clip(lam_left, -50 * rho_factor, 50 * rho_factor)
        lam_right = np.clip(lam_right, -50 * rho_factor, 50 * rho_factor)
        lam_bottom = np.clip(lam_bottom, -50 * rho_factor, 50 * rho_factor)
        lam_top = np.clip(lam_top, -50 * rho_factor, 50 * rho_factor)
        
        # Adaptive rho: increase if not converging
        if it % 50 == 49:
            max_viol = max(
                float(np.max(np.maximum(-g_cc, 0.0))) if len(g_cc) > 0 else 0,
                float(np.max(left_viol)) if len(left_viol) > 0 else 0,
                float(np.max(right_viol)) if len(right_viol) > 0 else 0,
                float(np.max(bottom_viol)) if len(bottom_viol) > 0 else 0,
                float(np.max(top_viol)) if len(top_viol) > 0 else 0
            )
            if max_viol > 0.01:
                rho_factor *= 1.5
                rho_cc = rho * rho_factor
                rho_wall = rho * rho_factor
            else:
                rho_factor = max(0.5, rho_factor * 0.9)
        
        # Track best
        if _is_valid(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_val:
                best_val = val
                best_state = (x.copy(), y.copy(), r.copy(), w)

    # Return best valid state
    if best_state is not None:
        x_best, y_best, r_best, w_best = best_state
        if _is_valid(x_best, y_best, r_best, w_best, 2.0 - w_best):
            z_out = np.empty(3 * N + 1)
            z_out[0::3][:n] = x_best
            z_out[1::3][:n] = y_best
            z_out[2::3][:n] = r_best
            z_out[-1] = w_best
            return z_out
    return None


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
        if time.time() > deadline - 20:
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

    # --- Phase 2: Polish on best result ---
    if best_result is not None and time.time() < deadline - 30:
        # 1. ADMM-based feasibility and optimization
        if time.time() < deadline - 25:
            admm_result = _admm_solve(best_result, deadline - (time.time() - start_time) - 20, rho=50.0)
            if admm_result is not None:
                valid, val = check_validity(admm_result)
                if valid and val > best_value:
                    best_value = val
                    best_result = admm_result

        # 2. Force-directed position polish to ensure we are in a good basin
        if best_result is not None and time.time() < deadline - 20:
            polished = _polish_result(best_result, deadline - (time.time() - start_time) - 15)
            if polished is not None:
                valid, val = check_validity(polished)
                if valid and val > best_value:
                    best_value = val
                    best_result = polished

        # 3. Expansion/Relocation Polish
        if best_result is not None and time.time() < deadline - 15:
            expanded = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 10)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # 4. Second ADMM pass with different rho
        if best_result is not None and time.time() < deadline - 10:
            admm_result2 = _admm_solve(best_result, deadline - (time.time() - start_time) - 8, rho=200.0)
            if admm_result2 is not None:
                valid, val = check_validity(admm_result2)
                if valid and val > best_value:
                    best_value = val
                    best_result = admm_result2

        # 5. Constraint-based SLSQP refinement
        if best_result is not None and time.time() < deadline - 8:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 5)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # 6. Final expansion pass
        if best_result is not None and time.time() < deadline - 4:
            expanded2 = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 2)
            if expanded2 is not None:
                valid, val = check_validity(expanded2)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded2

        # 7. Final quick polish
        if best_result is not None and time.time() < deadline - 2:
            polished2 = _polish_result(best_result, deadline - (time.time() - start_time))
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
        r_trial = r * (1 - 1e-8)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))


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


def _polish_result(z, time_budget):
    """Polish a valid packing with force-directed position adjustment and radius growth (vectorized)."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dt = 0.002
    max_iter = int(min(5000, time_budget * 200))
    prev_best = float(np.sum(r))

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
        fx -= left_viol * 40
        fx += right_viol * 40
        fy -= bottom_viol * 40
        fy += top_viol * 40

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

        if np.max(overlap) > 1e-12:
            f_pairs = overlap * 20.0
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
        elif max_force > 0.01:
            dt = 0.001
        else:
            dt = 0.002

        x_new = x + dt * fx
        y_new = y + dt * fy
        x_new = np.clip(x_new, r, w - r)
        y_new = np.clip(y_new, r, h - r)

        # Only accept if valid
        if _is_valid(x_new, y_new, r, w, h):
            x = x_new
            y = y_new

    # Try to grow radii uniformly with binary search
    for _ in range(100):
        if time.time() > time_budget:
            break
        # Binary search for max uniform growth
        lo, hi = 1.0, 1.05
        for _ in range(15):
            mid = (lo + hi) / 2.0
            if _is_valid(x, y, r * mid, w, h):
                lo = mid
            else:
                hi = mid
        if lo > 1.0 + 1e-9:
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
    Advanced polish: Iteratively expand circles and shift positions to create space.
    Improved: 8-direction moves, binary search for max radius, two expansion passes.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # 8 directions
    dirs = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (0.7071, 0.7071), (-0.7071, 0.7071), (0.7071, -0.7071), (-0.7071, -0.7071)
    ]

    def try_expand_circle(i):
        """Try to expand circle i to max possible radius."""
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
            for _ in range(25):
                mid = (lo + hi) / 2.0
                r_test = r.copy()
                r_test[i] = mid
                if _is_valid(x, y, r_test, w, h):
                    lo = mid
                else:
                    hi = mid
            if lo > r[i] + 1e-7:
                r[i] = lo
                return True
        return False

    def try_move_circle(i):
        """Try to move circle i in various directions to create space."""
        # Find the direction of greatest constraint
        best_dir = None
        best_score = -float('inf')
        
        for dx_d, dy_d in dirs:
            # Score: how much space would be created
            new_x = x[i] + dx_d * 0.01
            new_y = y[i] + dy_d * 0.01
            if new_x < r[i] or new_x > w - r[i] or new_y < r[i] or new_y > h - r[i]:
                continue
            
            # Check if this move creates more space
            min_gap = float('inf')
            for j in range(n):
                if i == j:
                    continue
                dx = new_x - x[j]
                dy = new_y - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                gap = d - (r[i] + r[j])
                if gap < min_gap:
                    min_gap = gap
            
            # Also consider wall distances
            wall_dist = min(new_x - r[i], w - new_x - r[i], new_y - r[i], h - new_y - r[i])
            score = min(min_gap, wall_dist)
            
            if score > best_score:
                best_score = score
                best_dir = (dx_d, dy_d)
        
        if best_dir is None:
            return False
        
        dx_d, dy_d = best_dir
        
        # Try multiple step sizes
        for step in [0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001]:
            new_x = x[i] + dx_d * step
            new_y = y[i] + dy_d * step
            if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                x[i] = new_x
                y[i] = new_y
                return True
        
        return False

    # Pass 1: Expand and move
    max_rounds = 40
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break
        
        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            
            # Try to expand
            if try_expand_circle(i):
                improved = True
            
            # If still limited, try to move
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
            
            if r_max - r[i] < 0.01:  # Limited expansion
                if try_move_circle(i):
                    improved = True
        
        if not improved:
            break

    # Pass 2: Second expansion after repositioning
    for rnd in range(20):
        if time.time() > time_budget:
            break
        
        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            if try_expand_circle(i):
                improved = True
        
        if not improved:
            break

    # Final uniform expansion with binary search
    for _ in range(100):
        if time.time() > time_budget:
            break
        lo, hi = 1.0, 1.05
        for _ in range(15):
            mid = (lo + hi) / 2.0
            if _is_valid(x, y, r * mid, w, h):
                lo = mid
            else:
                hi = mid
        if lo > 1.0 + 1e-9:
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
    """High-precision SLSQP refinement using explicit inequality constraints."""
    z = z.copy().astype(float)
    n = N
    n_pairs = n * (n - 1) // 2

    def obj_refine(zv):
        """Objective: minimize negative sum of radii."""
        r = zv[2::3][:n]
        return -np.sum(r)

    def make_circle_constraint(i):
        """Circle i must not overlap with any other circle."""
        def constraint(zv):
            x = zv[0::3][:n]
            y = zv[1::3][:n]
            r = zv[2::3][:n]
            # For each pair (i, j), enforce dist(i,j) >= r[i] + r[j]
            # Equivalent: dist^2 - (r[i]+r[j])^2 >= 0
            # But SLSQP needs g(z) >= 0, so we use dist - (r[i]+r[j]) >= 0
            # Actually, let's use the squared form for smoothness but it's not convex
            # Better: use dist - (r[i]+r[j]) >= 0 which is smooth when dist > 0
            
            dx = x[i] - x
            dy = y[i] - y
            dist = np.sqrt(dx * dx + dy * dy + 1e-12)
            # For all j != i: dist_j - (r[i] + r[j]) >= 0
            result = dist - (r[i] + r)
            result[i] = 1.0  # Self-constraint is always satisfied
            return result
        return constraint

    def make_wall_constraint(axis, side):
        """Wall constraint: circle must stay inside rectangle."""
        def constraint(zv):
            x = zv[0::3][:n]
            y = zv[1::3][:n]
            r = zv[2::3][:n]
            w = zv[-1]
            h = 2.0 - w
            
            if axis == 'x':
                pos = x
            else:
                pos = y
            
            if side == 'left' or side == 'bottom':
                # pos - r >= 0
                return pos - r
            else:
                # (w or h) - pos - r >= 0
                if axis == 'x':
                    return w - pos - r
                else:
                    return h - pos - r
        return constraint

    # Build constraints
    constraints = []
    
    # Circle-circle constraints (one per circle, each returning n-1 values)
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': make_circle_constraint(i)})
    
    # Wall constraints (4 per circle, but we can vectorize)
    # Left wall: x[i] - r[i] >= 0 for all i
    constraints.append({'type': 'ineq', 'fun': make_wall_constraint('x', 'left')})
    # Right wall: w - x[i] - r[i] >= 0 for all i
    constraints.append({'type': 'ineq', 'fun': make_wall_constraint('x', 'right')})
    # Bottom wall: y[i] - r[i] >= 0 for all i
    constraints.append({'type': 'ineq', 'fun': make_wall_constraint('y', 'bottom')})
    # Top wall: h - y[i] - r[i] >= 0 for all i
    constraints.append({'type': 'ineq', 'fun': make_wall_constraint('y', 'top')})

    bounds = []
    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    maxiter = int(min(1500, time_budget * 300))
    try:
        res = minimize(obj_refine, z, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 200})

        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w

        # Try to repair if slightly invalid
        if not _is_valid(x, y, r, w, h):
            # Shrink radii slightly
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
