<<<<<<< SEARCH
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
        h = z_opt[-1] # Wait, h is 2-w. 
        h = 2.0 - w

        if _is_valid(x, y, r, w, h):
            return z_opt
        return None
    except Exception:
        return None
=======
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


def _local_expansion_polish(z, time_budget):
    """
    Polish by attempting to expand each circle individually.
    If a circle is blocked by a neighbor or wall, try to move it away to create space.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    iu = np.triu_indices(n, k=1)
    
    # Precompute distances
    dx_all = x[:, None] - x[None, :]
    dy_all = y[:, None] - y[None, :]
    d_all = np.sqrt(dx_all * dx_all + dy_all * dy_all)
    np.fill_diagonal(d_all, np.inf)

    max_rounds = 20
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break
            
        improved = False
        # Try to expand each circle
        for i in range(n):
            # Calculate max radius for circle i
            # Limited by walls
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            # Limited by neighbors
            r_neigh = np.inf
            for j in range(n):
                if i == j: continue
                d = d_all[i, j]
                r_limit = d - r[j]
                if r_limit < r_neigh:
                    r_neigh = r_limit
            
            r_max = max(0, min(r_wall, r_neigh))
            
            # If we can expand, do it
            if r_max > r[i] + 1e-6:
                # Expand to 99% of max to be safe
                new_r = r[i] + (r_max - r[i]) * 0.99
                r_trial = r.copy()
                r_trial[i] = new_r
                if _is_valid(x, y, r_trial, w, h):
                    r[i] = new_r
                    improved = True
                    # Update distances
                    d_all[i, :] = np.sqrt((x[i] - x)**2 + (y[i] - y)**2)
                    d_all[:, i] = d_all[i, :]
                    d_all[i, i] = np.inf
                else:
                    # Try to move circle i to free up space
                    # Identify the constraint
                    # If r_max is determined by a neighbor j, move away from j
                    # If r_max is determined by a wall, move away from wall
                    
                    # Find tightest neighbor
                    min_gap = np.inf
                    tight_j = -1
                    for j in range(n):
                        if i == j: continue
                        gap = d_all[i, j] - (r[i] + r[j])
                        if gap < min_gap:
                            min_gap = gap
                            tight_j = j
                    
                    # Check walls
                    ml = x[i] - r[i]
                    mr = w - x[i] - r[i]
                    mb = y[i] - r[i]
                    mt = h - y[i] - r[i]
                    
                    # Determine direction to move
                    move_x = 0.0
                    move_y = 0.0
                    move_magnitude = 0.0
                    
                    # Compare gaps
                    if min_gap < ml and min_gap < mr and min_gap < mb and min_gap < mt:
                        # Move away from tight_j
                        if tight_j != -1:
                            dx = x[i] - x[tight_j]
                            dy = y[i] - y[tight_j]
                            d = np.sqrt(dx*dx + dy*dy)
                            if d > 1e-8:
                                move_x = dx / d
                                move_y = dy / d
                                move_magnitude = 0.05
                    elif ml <= mr and ml <= mb and ml <= mt:
                        move_x = 1.0
                        move_magnitude = 0.05
                    elif mr <= ml and mr <= mb and mr <= mt:
                        move_x = -1.0
                        move_magnitude = 0.05
                    elif mb <= ml and mb <= mr and mb <= mt:
                        move_y = 1.0
                        move_magnitude = 0.05
                    else:
                        move_y = -1.0
                        move_magnitude = 0.05
                    
                    # Try moving
                    if move_magnitude > 0:
                        new_x = x[i] + move_x * move_magnitude
                        new_y = y[i] + move_y * move_magnitude
                        
                        if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                            x[i] = new_x
                            y[i] = new_y
                            improved = True
                            # Update distances
                            d_all[i, :] = np.sqrt((x[i] - x)**2 + (y[i] - y)**2)
                            d_all[:, i] = d_all[i, :]
                            d_all[i, i] = np.inf
                            
        if not improved:
            break
            
    # Final uniform expansion
    grow_factor = 1.0005
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
>>>>>>> REPLACE

