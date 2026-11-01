"""Circle packing in perimeter-4 rectangle: SLSQP + Granular/Contact Polish + Epigraph Refine."""
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
        
        # 1. Hexagonal packing initial guesses (denser than square grid)
        for w0 in [1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 0.65]:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            
            z = np.zeros(3 * N + 1)
            # Simple hex layout scaled to fit
            grid_cols = 8
            grid_rows = 6
            x_spacing = w0 / (grid_cols + 0.5)
            y_spacing = x_spacing * np.sqrt(3) / 2.0
            
            total_height = grid_rows * y_spacing
            if total_height > h0:
                scale = h0 / total_height
                x_spacing *= scale
                y_spacing *= scale
            
            idx = 0
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if idx >= N:
                        break
                    cx = (col + 0.5) * x_spacing
                    cy = (row + 0.5) * y_spacing
                    if row % 2 == 1:
                        cx += x_spacing / 2.0
                    
                    if cx < 0 or cx > w0 or cy < 0 or cy > h0:
                        continue
                        
                    r0 = min(x_spacing, y_spacing) * 0.49
                    z[3 * idx] = cx
                    z[3 * idx + 1] = cy
                    z[3 * idx + 2] = r0
                    idx += 1
            
            if idx < N:
                for i in range(idx, N):
                    z[3 * i] = w0 / 2.0
                    z[3 * i + 1] = h0 / 2.0
                    z[3 * i + 2] = 0.05
                    
            z[-1] = w0
            guesses.append(z)

        # 2. Square grid initial guesses
        side = int(np.ceil(np.sqrt(N)))
        for w0 in [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8]:
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

        # 3. Random starts
        rng = np.random.default_rng(42)
        for _ in range(15):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.04, 0.14)
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
        z0_relaxed = _force_relax_unified(z0, max_iter=800)

        # SLSQP optimization
        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(800, max(200, remaining * 60)))
        try:
            res = minimize(objective, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 100})

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

    # --- Phase 2: Advanced Polish on best result ---
    if best_result is not None and time.time() < deadline - 30:
        # 1. Force-directed position polish (Granular model)
        polished = _granular_polish(best_result, deadline - (time.time() - start_time) - 20)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        # 2. Vectorized Expansion/Relocation Polish
        if best_result is not None and time.time() < deadline - 20:
            expanded = _expansion_relaxation_polish_v2(best_result, deadline - (time.time() - start_time) - 10)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # 3. Epigraph-based SLSQP refinement (Smooth constraints)
        if best_result is not None and time.time() < deadline - 15:
            refined = _epigraph_refine(best_result, deadline - (time.time() - start_time) - 5)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # 4. Final quick polish
        if best_result is not None and time.time() < deadline - 5:
            polished2 = _granular_polish(best_result, deadline - (time.time() - start_time))
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


def _granular_polish(z, time_budget):
    """
    Model the packing as a 2D granular material under uniform hydrostatic pressure.
    Circles expand until they reach a rigid contact equilibrium.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    iu = np.triu_indices(n, k=1)
    
    # Parameters
    pressure = 1.0  # Hydrostatic pressure coefficient
    max_iter = int(min(5000, time_budget * 250))
    dt = 0.001

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # 1. Wall pressure (Hydrostatic)
        # Force proportional to penetration
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        
        fx -= left_viol * pressure * 50
        fx += right_viol * pressure * 50
        fy -= bottom_viol * pressure * 50
        fy += top_viol * pressure * 50

        # 2. Pairwise contact forces
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) > 1e-10:
            # Hertzian-like or linear repulsion. Linear is stable.
            f_pairs = overlap * pressure * 20.0
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

        # Update positions
        x += dt * fx
        y += dt * fy
        
        # Clamp
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

        # 3. Attempt expansion (Volume increase)
        # If we have space, increase radius slightly.
        # Check if any circle has significant clearance.
        # We can just try to expand all by a tiny amount if valid.
        # This simulates the "pressure" pushing them out.
        growth = 1.0 + 1e-5
        r_trial = r * growth
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            # Try to expand only those with most clearance? 
            # For now, stop expanding if blocked.
            # This creates a "rigid contact equilibrium" where only non-contacted circles expand.
            # A more sophisticated approach would expand based on local pressure.
            pass

    # Final uniform expansion
    grow_factor = 1.0005
    for _ in range(500):
        if time.time() > time_budget:
            break
        r_trial = r * grow_factor
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
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


def _expansion_relaxation_polish_v2(z, time_budget):
    """
    Vectorized expansion polish: computes max radii for all circles simultaneously,
    then performs coordinated position shifts.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    iu = np.triu_indices(n, k=1)

    max_rounds = 60
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        # Vectorized: compute max possible radius for each circle
        dx_all = x[:, None] - x[None, :]
        dy_all = y[:, None] - y[None, :]
        d_all = np.sqrt(dx_all * dx_all + dy_all * dy_all)
        np.fill_diagonal(d_all, np.inf)

        # r_max[i] = min over j!=i of (d[i,j] - r[j]), also bounded by wall
        r_from_others = d_all - r[None, :]  # shape (n, n), r_from_others[i,j] = d[i,j] - r[j]
        np.fill_diagonal(r_from_others, np.inf)
        r_from_neigh = np.min(r_from_others, axis=1)  # shape (n,)

        r_wall = np.minimum(np.minimum(x, w - x), np.minimum(y, h - y))
        r_max = np.maximum(0.0, np.minimum(r_wall, r_from_neigh))

        # Expand all circles that can grow
        grow_mask = r_max > r + 1e-7
        if np.any(grow_mask):
            # Use 95% of available to stay safe
            r_new = np.where(grow_mask, np.minimum(r_max * 0.95, r + (r_max - r) * 0.5), r)
            # Verify validity
            if _is_valid(x, y, r_new, w, h):
                r = r_new
            else:
                # Fallback: expand one at a time
                for i in np.where(grow_mask)[0]:
                    r_trial = r.copy()
                    r_trial[i] = r_max[i] * 0.95
                    if _is_valid(x, y, r_trial, w, h):
                        r = r_trial

        # Vectorized: find circles that are "stuck" (close to constraint)
        margin_left = x - r
        margin_right = w - x - r
        margin_bottom = y - r
        margin_top = h - y - r

        # Pairwise gaps
        d_pairs = d_all[iu]
        r_sum_pairs = r[iu[0]] + r[iu[1]]
        gap_pairs = d_pairs - r_sum_pairs

        # For each circle, find minimum gap
        min_gap = np.full(n, 0.05)  # threshold
        for idx in range(len(iu[0])):
            i_idx = iu[0][idx]
            j_idx = iu[1][idx]
            g = gap_pairs[idx]
            if g < min_gap[i_idx]:
                min_gap[i_idx] = g
            if g < min_gap[j_idx]:
                min_gap[j_idx] = g

        min_margin = np.minimum(np.minimum(margin_left, margin_right),
                                np.minimum(margin_bottom, margin_top))
        min_constraint = np.minimum(min_gap, min_margin)

        stuck = min_constraint < 0.015
        if not np.any(stuck):
            break

        # Move stuck circles away from their tightest constraint
        improved = False
        for i in np.where(stuck)[0]:
            # Determine direction to move
            ml = margin_left[i]
            mr = margin_right[i]
            mb = margin_bottom[i]
            mt = margin_top[i]
            mg = min_gap[i]

            move_x = 0.0
            move_y = 0.0

            if mg <= ml and mg <= mr and mg <= mb and mg <= mt:
                # Nearest neighbor is the constraint
                # Find which neighbor
                best_j = -1
                best_d = float('inf')
                for j in range(n):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d < best_d:
                        best_d = d
                        best_j = j
                if best_j >= 0:
                    dx = x[i] - x[best_j]
                    dy = y[i] - y[best_j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d > 1e-8:
                        move_x = dx / d
                        move_y = dy / d
            elif ml <= mr and ml <= mb and ml <= mt:
                move_x = 1.0
            elif mr <= ml and mr <= mb and mr <= mt:
                move_x = -1.0
            elif mb <= ml and mb <= mr and mb <= mt:
                move_y = 1.0
            else:
                move_y = -1.0

            # Try moves
            for step in [0.04, 0.02, 0.01, 0.005, 0.002]:
                new_x = x[i] + move_x * step
                new_y = y[i] + move_y * step
                if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                    x[i] = new_x
                    y[i] = new_y
                    improved = True
                    break
            else:
                # Perpendicular
                perp_x = -move_y
                perp_y = move_x
                for step in [0.02, 0.01, 0.005]:
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

    # Final uniform expansion
    grow_factor = 1.001
    for _ in range(300):
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


def _epigraph_refine(z, time_budget):
    """
    High-precision SLSQP refinement using an epigraph formulation with smooth constraints.
    We introduce t_i as the radius, and use soft constraints for overlaps and walls.
    Objective: -sum(t_i)
    Constraints: 
    1. x_i - t_i >= 0
    2. w - x_i - t_i >= 0
    3. y_i - t_i >= 0
    4. h - y_i - t_i >= 0
    5. (x_i - x_j)^2 + (y_i - y_j)^2 - (t_i + t_j)^2 >= 0
    """
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

    maxiter = int(min(800, time_budget * 150))
    try:
        res = minimize(obj_refine, z, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})

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
