"""Circle packing in perimeter-4 rectangle: Rigid-body dynamics with adaptive collision resolution."""
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


def _surrogate_energy(cx, cy, r, W, H):
    """
    Surrogate energy: penalizes overlaps (quadratic) and wall violations.
    Lower is better.
    """
    energy = 0.0
    # Wall penalties (quadratic)
    wall_pen = 0.0
    for i in range(N):
        wall_pen += max(0.0, r[i] - cx[i]) ** 2
        wall_pen += max(0.0, cx[i] + r[i] - W) ** 2
        wall_pen += max(0.0, r[i] - cy[i]) ** 2
        wall_pen += max(0.0, cy[i] + r[i] - H) ** 2
    energy += wall_pen

    # Pairwise overlap penalties (quadratic)
    overlap_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d2 = dx * dx + dy * dy
            rsum = r[i] + r[j]
            if d2 < rsum * rsum:
                overlap = rsum - np.sqrt(max(d2, 1e-14))
                overlap_pen += overlap * overlap
    energy += 10.0 * overlap_pen  # Weight overlaps more heavily

    return energy


def _generate_hexagonal(w, h, n_circles, rng=None):
    """Generate hexagonal close-packed initial positions."""
    if rng is None:
        rng = np.random.default_rng()
    
    positions = []
    # Estimate optimal radius from area
    area_per_circle = (w * h) / n_circles
    r_est = np.sqrt(area_per_circle / (np.pi * 2 / np.sqrt(3)))
    r_est = min(r_est * 0.92, w / 5.5, h / 5.5)
    
    # Generate hexagonal lattice
    row_height = r_est * np.sqrt(3)
    col_width = 2 * r_est
    
    rows = int(h / row_height) + 2
    cols = int(w / col_width) + 2
    
    for row in range(rows):
        for col in range(cols):
            if len(positions) >= n_circles:
                break
            x = (col + 0.5) * col_width + (row % 2) * r_est
            y = (row + 0.5) * row_height
            if x + r_est <= w and y + r_est <= h and x - r_est >= 0 and y - r_est >= 0:
                positions.append((x, y, r_est))
    
    # Fill remaining with best clearance spots
    while len(positions) < n_circles:
        best_pos = None
        best_clearance = -1
        for _ in range(200):
            tx = rng.uniform(r_est * 0.5, w - r_est * 0.5)
            ty = rng.uniform(r_est * 0.5, h - r_est * 0.5)
            clearance = min(tx, w - tx, ty, h - ty)
            for px, py, pr in positions:
                d = np.sqrt((tx - px) ** 2 + (ty - py) ** 2)
                clearance = min(clearance, d - pr - r_est)
            if clearance > best_clearance:
                best_clearance = clearance
                best_pos = (tx, ty, r_est * 0.85)
        if best_pos is not None and best_clearance > 0:
            positions.append(best_pos)
        else:
            break
    
    return positions[:n_circles]


def _generate_grid(w, h, n_circles, rng=None):
    """Generate grid-based initial positions."""
    positions = []
    best_config = None
    best_score = 0
    
    for rows in range(3, 9):
        for cols in range(3, 9):
            if rows * cols < n_circles:
                continue
            cell_w = w / cols
            cell_h = h / rows
            r_est = min(cell_w, cell_h) / 2 * 0.93
            
            # Check fit
            valid = True
            for i in range(n_circles):
                row = i // cols
                col = i % cols
                if row >= rows:
                    valid = False
                    break
                x = (col + 0.5) * cell_w
                y = (row + 0.5) * cell_h
                if x - r_est < 0 or x + r_est > w or y - r_est < 0 or y + r_est > h:
                    valid = False
                    break
            
            if valid:
                # Score by minimum inter-circle distance
                min_d = float('inf')
                for i in range(n_circles):
                    row_i = i // cols
                    col_i = i % cols
                    xi = (col_i + 0.5) * cell_w
                    yi = (row_i + 0.5) * cell_h
                    for j in range(i + 1, n_circles):
                        row_j = j // cols
                        col_j = j % cols
                        xj = (col_j + 0.5) * cell_w
                        yj = (row_j + 0.5) * cell_h
                        d = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                        min_d = min(min_d, d)
                score = min_d / r_est if r_est > 0 else 0
                if score > best_score:
                    best_score = score
                    best_config = (rows, cols, r_est)
    
    if best_config is None:
        # Fallback
        cols = n_circles
        cell_w = w / cols
        r_est = cell_w / 2 * 0.88
        for i in range(n_circles):
            x = (i + 0.5) * cell_w
            y = h / 2
            positions.append((x, y, r_est))
    else:
        rows, cols, r_est = best_config
        cell_w = w / cols
        cell_h = h / rows
        for i in range(n_circles):
            row = i // cols
            col = i % cols
            x = (col + 0.5) * cell_w
            y = (row + 0.5) * cell_h
            positions.append((x, y, r_est))
    
    return positions[:n_circles]


def _dynamics_solve(x, y, r, w, h, time_budget, dt=0.0008, max_iter=50000):
    """
    Rigid-body dynamics with adaptive time-stepping and impulse collision resolution.
    Mass proportional to radius^2.
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    
    if w < 0.15 or h < 0.15:
        return None, None, None

    vx = np.zeros(N)
    vy = np.zeros(N)
    mass = np.maximum(r * r, 1e-8)

    # Clamp initial positions
    x = np.clip(x, r, w - r)
    y = np.clip(y, r, h - r)

    damping = 0.90
    start_time = time.time()
    
    # Force parameters
    wall_stiffness = 8000.0
    pair_stiffness = 5000.0

    iu = np.triu_indices(N, k=1)

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

        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-14))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        max_overlap = np.max(overlap) if len(overlap) > 0 else 0.0

        if max_overlap > 1e-12:
            f_pairs = overlap * pair_stiffness
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            np.add.at(fx, iu[0], ux * f_pairs)
            np.add.at(fy, iu[0], uy * f_pairs)
            np.add.at(fx, iu[1], -ux * f_pairs)
            np.add.at(fy, iu[1], -uy * f_pairs)

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

        # Impulse-based collision resolution - multiple passes
        for _ in range(10):
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d_sq = dx * dx + dy * dy
            r_sum = r[:, None] + r[None, :]

            d_sq_pairs = d_sq[iu]
            r_sum_pairs = r_sum[iu]
            d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-14))
            overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

            if np.max(overlap) <= 1e-10:
                break

            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            m_i = mass[iu[0]]
            m_j = mass[iu[1]]
            total_m = m_i + m_j
            correction = overlap * 0.5
            frac_i = m_j / total_m
            frac_j = m_i / total_m

            dx_corr = ux * correction * frac_i
            dy_corr = uy * correction * frac_i
            np.add.at(x, iu[0], dx_corr)
            np.add.at(y, iu[0], dy_corr)
            np.add.at(x, iu[1], -ux * correction * frac_j)
            np.add.at(y, iu[1], -uy * correction * frac_j)

            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)

        # Early exit if converged
        if max_overlap < 1e-8 and np.max(np.abs(vx)) < 1e-6 and np.max(np.abs(vy)) < 1e-6:
            break

    if not _is_valid(x, y, r, w, h):
        return None, None, None

    return x, y, r


def _expand_and_move(x, y, r, w, h, time_budget):
    """
    Joint optimization: expand radii and move circles to create space.
    Uses iterative local search with greedy improvement.
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)

    start_time = time.time()
    improved = True
    rounds = 0

    while improved and time.time() - start_time < time_budget and rounds < 500:
        improved = False
        rounds += 1

        # Phase 1: Individual radius expansion via binary search
        for i in range(N):
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
            r_max = max(0.0, r_max)

            if r_max > r[i] + 1e-8:
                lo = r[i]
                hi = r_max
                for _ in range(40):
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-9:
                    r[i] = lo
                    improved = True

        # Phase 2: Move circles to create expansion space
        for i in range(N):
            best_move = None
            best_gain = 0.0
            best_x, best_y = x[i], y[i]

            # Try 36 directions with multiple step sizes
            for angle_idx in range(36):
                angle = angle_idx * (2 * np.pi / 36)
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                for step in [0.05, 0.03, 0.015, 0.008, 0.004, 0.002]:
                    new_x = x[i] + cos_a * step
                    new_y = y[i] + sin_a * step

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
                        if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - 1e-10:
                            valid = False
                            break

                    if valid:
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

            if best_move is not None and best_gain > 1e-7:
                x[i] = best_move[0]
                y[i] = best_move[1]
                improved = True

        # Phase 3: Pairwise moves to create space
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                if d < 1e-8:
                    continue
                ux = dx / d
                uy = dy / d

                for step in [0.012, 0.006, 0.003, 0.0015]:
                    new_xi = x[i] + ux * step
                    new_yi = y[i] + uy * step
                    new_xj = x[j] - ux * step
                    new_yj = y[j] - uy * step

                    if new_xi < r[i] or new_xi > w - r[i] or new_yi < r[i] or new_yi > h - r[i]:
                        continue
                    if new_xj < r[j] or new_xj > w - r[j] or new_yj < r[j] or new_yj > h - r[j]:
                        continue

                    valid = True
                    for k in range(N):
                        if k == i or k == j:
                            continue
                        dx_ik = new_xi - x[k]
                        dy_ik = new_yi - y[k]
                        if dx_ik * dx_ik + dy_ik * dy_ik < (r[i] + r[k]) ** 2 - 1e-10:
                            valid = False
                            break
                        dx_jk = new_xj - x[k]
                        dy_jk = new_yj - y[k]
                        if dx_jk * dx_jk + dy_jk * dy_jk < (r[j] + r[k]) ** 2 - 1e-10:
                            valid = False
                            break

                    if valid:
                        x_test = x.copy()
                        y_test = y.copy()
                        x_test[i] = new_xi
                        y_test[i] = new_yi
                        x_test[j] = new_xj
                        y_test[j] = new_yj

                        r_test = r.copy()
                        for _ in range(15):
                            r_trial = r_test * 1.004
                            if _is_valid(x_test, y_test, r_trial, w, h):
                                r_test = r_trial
                            else:
                                break

                        if np.sum(r_test) > np.sum(r) + 1e-7:
                            x[i] = new_xi
                            y[i] = new_yi
                            x[j] = new_xj
                            y[j] = new_yj
                            r = r_test
                            improved = True
                            break

    # Final uniform scaling
    for _ in range(8000):
        if time.time() - start_time > time_budget:
            break
        r_trial = r * 1.0002
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    # Second individual expansion pass
    for i in range(N):
        if time.time() - start_time > time_budget:
            break
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
        r_max = max(0.0, r_max)
        if r_max > r[i] + 1e-8:
            lo = r[i]
            hi = r_max
            for _ in range(35):
                mid = (lo + hi) / 2.0
                r_test = r.copy()
                r_test[i] = mid
                if _is_valid(x, y, r_test, w, h):
                    lo = mid
                else:
                    hi = mid
            if lo > r[i] + 1e-9:
                r[i] = lo

    if _is_valid(x, y, r, w, h):
        return x, y, r
    return None, None, None


def _optimize_width(x, y, r, w0, h0, time_budget):
    """
    Optimize rectangle width (aspect ratio) with multi-scale search.
    """
    best_w = w0
    best_val = np.sum(r)
    start_time = time.time()

    # Coarse search
    for w in np.linspace(0.25, 1.75, 70):
        if time.time() - start_time > time_budget:
            break
        h = 2.0 - w
        if h < 0.2:
            continue
        scale_x = w / w0
        scale_y = h / h0
        x_new = x * scale_x
        y_new = y * scale_y
        r_new = r * min(scale_x, scale_y)
        if _is_valid(x_new, y_new, r_new, w, h):
            val = np.sum(r_new)
            if val > best_val:
                best_val = val
                best_w = w

    # Fine search
    for w in np.linspace(max(0.2, best_w - 0.15), min(1.8, best_w + 0.15), 40):
        if time.time() - start_time > time_budget:
            break
        h = 2.0 - w
        if h < 0.2:
            continue
        scale_x = w / w0
        scale_y = h / h0
        x_new = x * scale_x
        y_new = y * scale_y
        r_new = r * min(scale_x, scale_y)
        if _is_valid(x_new, y_new, r_new, w, h):
            val = np.sum(r_new)
            if val > best_val:
                best_val = val
                best_w = w

    # Ultra-fine search
    for w in np.linspace(max(0.2, best_w - 0.04), min(1.8, best_w + 0.04), 25):
        if time.time() - start_time > time_budget:
            break
        h = 2.0 - w
        if h < 0.2:
            continue
        scale_x = w / w0
        scale_y = h / h0
        x_new = x * scale_x
        y_new = y * scale_y
        r_new = r * min(scale_x, scale_y)
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
    rng = np.random.default_rng(12345)

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

    # Generate diverse initial guesses
    def get_initial_guesses():
        guesses = []
        aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 0.65, 1.4, 0.6, 1.5, 0.55, 1.15, 0.85, 0.95, 1.05, 0.75, 1.25, 0.72, 1.28, 0.68, 1.32, 0.58, 1.42, 0.52, 1.48]

        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.35:
                continue

            # Hexagonal
            positions = _generate_hexagonal(w0, h0, N, rng)
            if len(positions) >= N:
                x = np.array([p[0] for p in positions])
                y = np.array([p[1] for p in positions])
                r = np.array([p[2] for p in positions])
                guesses.append((x, y, r, w0, h0))

            # Grid
            positions = _generate_grid(w0, h0, N, rng)
            if len(positions) >= N:
                x = np.array([p[0] for p in positions])
                y = np.array([p[1] for p in positions])
                r = np.array([p[2] for p in positions])
                guesses.append((x, y, r, w0, h0))

        # Random starts
        for trial in range(80):
            w0 = rng.uniform(0.4, 1.6)
            h0 = 2.0 - w0
            if h0 < 0.35:
                continue
            x = rng.uniform(0.05, w0 - 0.05, N)
            y = rng.uniform(0.05, h0 - 0.05, N)
            r_base = rng.uniform(0.04, 0.12)
            r = r_base * rng.uniform(0.7, 1.3, N)
            guesses.append((x, y, r, w0, h0))

        return guesses

    guesses = get_initial_guesses()

    # Sort by surrogate energy
    def guess_energy(g):
        x, y, r, w, h = g
        return _surrogate_energy(x, y, r, w, h)

    guesses.sort(key=guess_energy)

    # Process top candidates
    max_candidates = min(len(guesses), 40)
    for i in range(max_candidates):
        if time.time() > deadline - 20:
            break

        x0, y0, r0, w0, h0 = guesses[i]

        # Step 1: Dynamics simulation
        remaining = deadline - (time.time() - start_time)
        sim_time = min(5.0, remaining * 0.22)
        x_sim, y_sim, r_sim = _dynamics_solve(x0, y0, r0, w0, h0, sim_time)
        if x_sim is None:
            continue

        # Step 2: Optimize aspect ratio
        remaining = deadline - (time.time() - start_time)
        aspect_time = min(2.5, remaining * 0.1)
        w_opt = _optimize_width(x_sim, y_sim, r_sim, w0, h0, aspect_time)
        if abs(w_opt - w0) > 1e-6:
            h_opt = 2.0 - w_opt
            scale_x = w_opt / w0
            scale_y = h_opt / h0
            x_sim = x_sim * scale_x
            y_sim = y_sim * scale_y
            r_sim = r_sim * min(scale_x, scale_y)
            w0 = w_opt
            h0 = h_opt

        # Step 3: Expand and move
        remaining = deadline - (time.time() - start_time)
        expand_time = min(18.0, remaining * 0.5)
        x_exp, y_exp, r_exp = _expand_and_move(x_sim, y_sim, r_sim, w0, h0, expand_time)
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

    # Final polish
    if best_result is not None and time.time() < deadline - 3:
        x_best, y_best, r_best, w_best, h_best = best_result
        final_time = max(2.0, deadline - (time.time() - start_time) - 2)
        x_final, y_final, r_final = _expand_and_move(x_best, y_best, r_best, w_best, h_best, final_time)
        if x_final is not None:
            valid, val = check_validity(x_final, y_final, r_final, w_best, h_best)
            if valid and val > best_value:
                best_value = val
                best_result = (x_final.copy(), y_final.copy(), r_final.copy(), w_best, h_best)

        # Aspect re-optimization
        if time.time() < deadline - 2:
            x_b, y_b, r_b, w_b, h_b = best_result
            w_opt2 = _optimize_width(x_b, y_b, r_b, w_b, h_b, 1.5)
            if abs(w_opt2 - w_b) > 1e-6:
                h_opt2 = 2.0 - w_opt2
                scale_x = w_opt2 / w_b
                scale_y = h_opt2 / h_b
                x_b2 = x_b * scale_x
                y_b2 = y_b * scale_y
                r_b2 = r_b * min(scale_x, scale_y)
                valid2, val2 = check_validity(x_b2, y_b2, r_b2, w_opt2, h_opt2)
                if valid2 and val2 > best_value:
                    best_value = val2
                    best_result = (x_b2.copy(), y_b2.copy(), r_b2.copy(), w_opt2, h_opt2)

        # Final expansion
        if time.time() < deadline - 1:
            x_b, y_b, r_b, w_b, h_b = best_result
            polish_time = max(1.0, deadline - (time.time() - start_time) - 0.5)
            x_p, y_p, r_p = _expand_and_move(x_b, y_b, r_b, w_b, h_b, polish_time)
            if x_p is not None:
                valid_p, val_p = check_validity(x_p, y_p, r_p, w_b, h_b)
                if valid_p and val_p > best_value:
                    best_value = val_p
                    best_result = (x_p.copy(), y_p.copy(), r_p.copy(), w_b, h_b)

    if best_result is None:
        # Fallback
        side = 5
        w = 1.0
        h = 1.0
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x_pos = (i + 0.5) / side
                y_pos = (j + 0.5) / side
                r_pos = 0.5 / side - 0.005
                cells.append((x_pos, y_pos, r_pos))
        return np.array(cells[:N])

    x, y, r, w, h = best_result

    # Safety shrink
    for _ in range(30):
        r_trial = r * (1 - 1e-8)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
