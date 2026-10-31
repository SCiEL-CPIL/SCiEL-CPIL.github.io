"""Circle packing in perimeter-4 rectangle: Rigid-body dynamics + surrogate optimization."""
import numpy as np
import time

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


def _hexagonal_packing(w, h, n_circles):
    """Generate initial positions using hexagonal close packing."""
    positions = []
    # Estimate radius for hexagonal packing
    area_per_circle = (w * h) / n_circles
    r_est = np.sqrt(area_per_circle / (np.pi * 2 / np.sqrt(3)))
    r_est = min(r_est * 0.9, w / 6, h / 6)
    
    rows = int(h / (r_est * np.sqrt(3))) + 1
    cols = int(w / (r_est * 2)) + 1
    
    for row in range(rows):
        for col in range(cols):
            if len(positions) >= n_circles:
                break
            x = (col + 0.5) * 2 * r_est + (row % 2) * r_est
            y = (row + 0.5) * r_est * np.sqrt(3)
            if x + r_est <= w and y + r_est <= h:
                positions.append((x, y, r_est))
    
    # If not enough, add remaining with smaller radius
    while len(positions) < n_circles:
        # Find a spot with maximum clearance
        best_pos = None
        best_clearance = 0
        for _ in range(100):
            tx = np.random.uniform(r_est, w - r_est)
            ty = np.random.uniform(r_est, h - r_est)
            clearance = min(tx, w - tx, ty, h - ty)
            for px, py, pr in positions:
                d = np.sqrt((tx - px) ** 2 + (ty - py) ** 2)
                clearance = min(clearance, d - pr - r_est)
            if clearance > best_clearance:
                best_clearance = clearance
                best_pos = (tx, ty, r_est * 0.8)
        if best_pos is not None and best_clearance > 0:
            positions.append(best_pos)
        else:
            break
    
    return positions[:n_circles]


def _dynamics_simulate(x, y, r, w, h, time_budget, dt=0.001, max_iter=20000):
    """
    Rigid-body dynamics simulation with impulse-based collision resolution.
    Mass proportional to radius squared.
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    
    if w < 0.2 or h < 0.2:
        return None, None, None

    vx = np.zeros(N)
    vy = np.zeros(N)
    mass = np.maximum(r * r, 1e-6)

    # Ensure valid start
    x = np.clip(x, r, w - r)
    y = np.clip(y, r, h - r)

    damping = 0.99
    start_time = time.time()
    
    # Adaptive force scaling
    wall_stiffness = 1000.0
    pair_stiffness = 500.0

    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break

        fx = np.zeros(N)
        fy = np.zeros(N)

        # Wall forces
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * wall_stiffness
        fx += right_viol * wall_stiffness
        fy -= bottom_viol * wall_stiffness
        fy += top_viol * wall_stiffness

        # Pairwise repulsion
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(N, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) > 1e-10:
            f_pairs = overlap * pair_stiffness
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

        # Update velocities
        vx += dt * fx / mass
        vy += dt * fy / mass

        # Damping
        vx *= damping
        vy *= damping

        # Update positions
        x += dt * vx
        y += dt * vy

        # Clamp to walls
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

        # Impulse-based collision resolution (post-step correction)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(N, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) > 1e-9:
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            for idx in range(len(iu[0])):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                m_i = mass[i_idx]
                m_j = mass[j_idx]
                total_m = m_i + m_j
                correction = overlap[idx] * 0.5
                x[i_idx] += ux[idx] * correction * (m_j / total_m)
                y[i_idx] += uy[idx] * correction * (m_j / total_m)
                x[j_idx] -= ux[idx] * correction * (m_i / total_m)
                y[j_idx] -= uy[idx] * correction * (m_i / total_m)

            # Re-clamp after correction
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)

    # Final validity check
    if not _is_valid(x, y, r, w, h):
        return None, None, None

    return x, y, r


def _expand_radii(x, y, r, w, h, time_budget):
    """
    Iteratively expand radii using binary search, moving circles to make space.
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)

    start_time = time.time()
    improved = True
    rounds = 0

    while improved and time.time() - start_time < time_budget and rounds < 100:
        improved = False
        rounds += 1

        for i in range(N):
            # Max radius limited by walls
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall

            # Max radius limited by neighbors
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

            if r_max > r[i] + 1e-7:
                # Binary search for max valid radius
                lo = r[i]
                hi = r_max
                for _ in range(30):
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h):
                        lo = mid
                    else:
                        hi = mid

                if lo > r[i] + 1e-8:
                    r[i] = lo
                    improved = True

        # If no improvement, try to move circles to create space
        if not improved:
            for i in range(N):
                # Find direction of least resistance
                best_move = None
                best_gain = 0.0

                # Try 12 directions
                for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                    for step in [0.03, 0.015, 0.008, 0.004]:
                        new_x = x[i] + np.cos(angle) * step
                        new_y = y[i] + np.sin(angle) * step

                        if new_x < r[i] or new_x > w - r[i]:
                            continue
                        if new_y < r[i] or new_y > h - r[i]:
                            continue

                        valid = True
                        for j in range(N):
                            if i == j:
                                continue
                            dx = new_x - x[j]
                            dy = new_y - y[j]
                            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - 1e-9:
                                valid = False
                                break

                        if valid:
                            # Estimate gain: how much can we expand after moving?
                            r_wall = min(new_x, w - new_x, new_y, h - new_y)
                            r_max = r_wall
                            for j in range(N):
                                if i == j:
                                    continue
                                dx = new_x - x[j]
                                dy = new_y - y[j]
                                d = np.sqrt(dx * dx + dy * dy)
                                r_allowed = d - r[j]
                                if r_allowed < r_max:
                                    r_max = r_allowed
                            gain = max(0.0, r_max - r[i])
                            if gain > best_gain:
                                best_gain = gain
                                best_move = (new_x, new_y)

                if best_move is not None and best_gain > 1e-6:
                    x[i] = best_move[0]
                    y[i] = best_move[1]
                    improved = True

    # Final uniform expansion - try scaling all radii
    for _ in range(500):
        if time.time() - start_time > time_budget:
            break
        r_trial = r * 1.001
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    if _is_valid(x, y, r, w, h):
        return x, y, r
    return None, None, None


def _optimize_aspect(x, y, r, w0, h0, time_budget):
    """
    Optimize aspect ratio (width) by trying different widths.
    """
    best_w = w0
    best_val = np.sum(r)
    
    # Try a range of aspect ratios
    for w in np.linspace(0.4, 1.6, 25):
        h = 2.0 - w
        if h < 0.3:
            continue
        
        # Scale positions to fit new rectangle
        scale_x = w / w0
        scale_y = h / h0
        x_new = x * scale_x
        y_new = y * scale_y
        r_new = r * min(scale_x, scale_y)
        
        # Quick validity check
        if _is_valid(x_new, y_new, r_new, w, h):
            val = np.sum(r_new)
            if val > best_val:
                best_val = val
                best_w = w
    
    return best_w


def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    best_value = 0.0
    best_result = None

    def check_validity(x, y, r, w, h, tol=1e-9):
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

    # Generate initial guesses using hexagonal packing
    def get_initial_guesses():
        guesses = []
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 1.25, 0.7]
        
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            
            positions = _hexagonal_packing(w0, h0, N)
            if len(positions) >= N:
                x = np.array([p[0] for p in positions])
                y = np.array([p[1] for p in positions])
                r = np.array([p[2] for p in positions])
                guesses.append((x, y, r, w0, h0))
        
        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(15):
            w0 = rng.uniform(0.5, 1.5)
            h0 = 2.0 - w0
            x = rng.uniform(0.08, w0 - 0.08, N)
            y = rng.uniform(0.08, h0 - 0.08, N)
            r = rng.uniform(0.06, 0.12, N)
            guesses.append((x, y, r, w0, h0))

        return guesses

    guesses = get_initial_guesses()

    for i, (x0, y0, r0, w0, h0) in enumerate(guesses):
        if time.time() > deadline - 25:
            break

        # Step 1: Dynamics simulation to resolve overlaps
        sim_time = min(4.0, (deadline - (time.time() - start_time)) * 0.3)
        x_sim, y_sim, r_sim = _dynamics_simulate(x0, y0, r0, w0, h0, sim_time)
        if x_sim is None:
            continue

        # Step 2: Optimize aspect ratio
        w_opt = _optimize_aspect(x_sim, y_sim, r_sim, w0, h0, 1.0)
        if abs(w_opt - w0) > 1e-6:
            h_opt = 2.0 - w_opt
            scale_x = w_opt / w0
            scale_y = h_opt / h0
            x_sim = x_sim * scale_x
            y_sim = y_sim * scale_y
            r_sim = r_sim * min(scale_x, scale_y)
            w0 = w_opt
            h0 = h_opt

        # Step 3: Expand radii
        expand_time = min(6.0, (deadline - (time.time() - start_time)) * 0.5)
        x_exp, y_exp, r_exp = _expand_radii(x_sim, y_sim, r_sim, w0, h0, expand_time)
        if x_exp is not None:
            valid, val = check_validity(x_exp, y_exp, r_exp, w0, h0)
            if valid and val > best_value:
                best_value = val
                best_result = (x_exp.copy(), y_exp.copy(), r_exp.copy(), w0, h0)
        else:
            valid, val = check_validity(x_sim, y_sim, r_sim, w0, h0)
            if valid and val > best_value:
                best_value = val
                best_result = (x_sim.copy(), y_sim.copy(), r_sim.copy(), w0, h0)

    # Final polish on best result
    if best_result is not None and time.time() < deadline - 5:
        x_best, y_best, r_best, w_best, h_best = best_result
        final_time = deadline - (time.time() - start_time) - 2
        x_final, y_final, r_final = _expand_radii(x_best, y_best, r_best, w_best, h_best, final_time)
        if x_final is not None:
            valid, val = check_validity(x_final, y_final, r_final, w_best, h_best)
            if valid and val > best_value:
                best_value = val
                best_result = (x_final.copy(), y_final.copy(), r_final.copy(), w_best, h_best)

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, y, r, w, h = best_result

    # Final safety shrink
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
