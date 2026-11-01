"""Circle packing in perimeter-4 rectangle: Rigid-body dynamics + surrogate optimization."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

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


def _compute_energy(cx, cy, r, W, H):
    """Compute surrogate energy: penalizes overlaps and wall proximity."""
    energy = 0.0
    # Wall penalties
    for i in range(N):
        energy += max(0, r[i] - cx[i]) ** 2
        energy += max(0, cx[i] + r[i] - W) ** 2
        energy += max(0, r[i] - cy[i]) ** 2
        energy += max(0, cy[i] + r[i] - H) ** 2
    # Pairwise overlap penalties
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d2 = dx * dx + dy * dy
            rsum = r[i] + r[j]
            if d2 < rsum * rsum:
                overlap = rsum - np.sqrt(max(d2, 1e-12))
                energy += overlap * overlap
    return energy


def _grid_packing(w, h, n_circles):
    """Generate initial positions using a grid layout with uniform radii."""
    positions = []
    # Try different grid configurations
    best_config = None
    best_min_dist = 0
    
    for rows in range(3, 8):
        for cols in range(3, 8):
            if rows * cols < n_circles:
                continue
            cell_w = w / cols
            cell_h = h / rows
            # Max radius is half the minimum cell dimension
            r_est = min(cell_w, cell_h) / 2 * 0.95
            
            # Check if all circles fit
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
                # Compute minimum distance between any two circles
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
                
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_config = (rows, cols, r_est)
    
    if best_config is None:
        # Fallback: single row
        cols = n_circles
        cell_w = w / cols
        r_est = cell_w / 2 * 0.9
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


def _dynamics_simulate(x, y, r, w, h, time_budget, dt=0.002, max_iter=50000):
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

    damping = 0.88
    start_time = time.time()
    
    # Adaptive force scaling
    wall_stiffness = 8000.0
    pair_stiffness = 5000.0

    iu = np.triu_indices(N, k=1)
    n_pairs = len(iu[0])

    # Softening parameter to allow initial overlaps to resolve more smoothly
    current_dt = dt
    converged = False

    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        # Anneal: start with softer forces, become stiffer
        progress = it / max_iter
        soft_factor = 1.0 - 0.7 * progress
        
        fx = np.zeros(N)
        fy = np.zeros(N)

        # Wall forces
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * wall_stiffness * soft_factor
        fx += right_viol * wall_stiffness * soft_factor
        fy -= bottom_viol * wall_stiffness * soft_factor
        fy += top_viol * wall_stiffness * soft_factor

        # Pairwise repulsion - vectorized
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        max_overlap = np.max(overlap)
        
        if max_overlap > 1e-12:
            # Force is proportional to overlap with quadratic component for stability
            f_pairs = overlap * pair_stiffness * soft_factor + overlap * overlap * 2000.0
            
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            # Vectorized force accumulation using np.add.at
            np.add.at(fx, iu[0], ux * f_pairs)
            np.add.at(fy, iu[0], uy * f_pairs)
            np.add.at(fx, iu[1], -ux * f_pairs)
            np.add.at(fy, iu[1], -uy * f_pairs)

        # Update velocities with adaptive dt
        # Scale dt based on max force to maintain stability
        max_force = np.max(np.sqrt(fx*fx + fy*fy)) if np.any(fx != 0) else 0
        adaptive_dt = current_dt * min(1.0, 500.0 / (max_force + 1.0))
        
        vx += adaptive_dt * fx / mass
        vy += adaptive_dt * fy / mass

        # Damping
        vx *= damping
        vy *= damping

        # Update positions
        x += adaptive_dt * vx
        y += adaptive_dt * vy

        # Clamp to walls
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

        # Impulse-based collision resolution (post-step correction) - multiple passes
        # Increase passes as we get closer to convergence
        n_passes = 12 + int(8 * progress)
        
        prev_max_overlap = max_overlap
        for _ in range(n_passes):
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d_sq = dx * dx + dy * dy
            r_sum = r[:, None] + r[None, :]

            d_sq_pairs = d_sq[iu]
            r_sum_pairs = r_sum[iu]
            d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
            overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
            curr_max_overlap = np.max(overlap)

            if curr_max_overlap <= 1e-9:
                converged = True
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

            # Re-clamp after correction
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)

            if curr_max_overlap < prev_max_overlap * 0.01:
                break
            prev_max_overlap = curr_max_overlap

        # Check for convergence: if max overlap is very small and not changing
        if max_overlap < 1e-8 and it > 100:
            # Do a final check
            if _is_valid(x, y, r, w, h):
                break

    # Final validity check
    if not _is_valid(x, y, r, w, h):
        return None, None, None

    return x, y, r


def _expand_radii(x, y, r, w, h, time_budget):
    """
    Iteratively expand radii using binary search, moving circles to make space.
    Uses joint optimization of positions and radii.
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

        # Phase 1: Try to expand each circle individually
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
                for _ in range(35):
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

        # Phase 2: Try to move circles to create space using gradient descent
        if time.time() - start_time < time_budget:
            # Use scipy to optimize positions to maximize minimum clearance
            def objective(params):
                """Negative of minimum clearance (we want to maximize clearance)."""
                px = params[:N]
                py = params[N:2*N]
                
                # Penalize wall violations
                for i in range(N):
                    if px[i] < r[i] or px[i] > w - r[i] or py[i] < r[i] or py[i] > h - r[i]:
                        return 100.0
                
                # Compute minimum clearance
                min_clear = float('inf')
                for i in range(N):
                    # Wall clearance
                    min_clear = min(min_clear, px[i] - r[i], w - px[i] - r[i], py[i] - r[i], h - py[i] - r[i])
                    # Neighbor clearance
                    for j in range(i + 1, N):
                        dx = px[i] - px[j]
                        dy = py[i] - py[j]
                        d = np.sqrt(dx * dx + dy * dy)
                        min_clear = min(min_clear, d - r[i] - r[j])
                
                return -min_clear

            x0_params = np.concatenate([x, y])
            bounds = [(r[i], w - r[i]) for i in range(N)] + [(r[i], h - r[i]) for i in range(N)]
            
            try:
                result = minimize(objective, x0_params, method='Nelder-Mead', 
                                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
                if result.fun < -1e-6:  # Found improvement
                    new_x = result.x[:N]
                    new_y = result.x[N:2*N]
                    if _is_valid(new_x, new_y, r, w, h):
                        # Check if this allows radius expansion
                        r_test = r * 1.002
                        if _is_valid(new_x, new_y, r_test, w, h):
                            x = new_x
                            y = new_y
                            improved = True
            except:
                pass

        # Phase 3: Local search - try moving individual circles
        if not improved and time.time() - start_time < time_budget:
            for i in range(N):
                # Find direction of least resistance
                best_move = None
                best_gain = 0.0

                # Try more directions with finer steps
                for angle in np.linspace(0, 2 * np.pi, 64, endpoint=False):
                    for step in [0.05, 0.03, 0.015, 0.008, 0.004, 0.002, 0.001, 0.0005]:
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

        # Phase 4: Joint optimization - try small simultaneous moves
        if not improved and time.time() - start_time < time_budget:
            # Try moving pairs of circles to create space
            for i in range(N):
                for j in range(i + 1, N):
                    # Try moving both circles slightly apart
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d < 1e-6:
                        continue
                    ux = dx / d
                    uy = dy / d
                    
                    for step in [0.015, 0.008, 0.004, 0.002, 0.001, 0.0005]:
                        new_xi = x[i] + ux * step
                        new_yi = y[i] + uy * step
                        new_xj = x[j] - ux * step
                        new_yj = y[j] - uy * step
                        
                        if new_xi < r[i] or new_xi > w - r[i]:
                            continue
                        if new_yi < r[i] or new_yi > h - r[i]:
                            continue
                        if new_xj < r[j] or new_xj > w - r[j]:
                            continue
                        if new_yj < r[j] or new_yj > h - r[j]:
                            continue
                        
                        # Check validity
                        valid = True
                        for k in range(N):
                            if k == i or k == j:
                                continue
                            dx_ik = new_xi - x[k]
                            dy_ik = new_yi - y[k]
                            if dx_ik * dx_ik + dy_ik * dy_ik < (r[i] + r[k]) ** 2 - 1e-9:
                                valid = False
                                break
                            dx_jk = new_xj - x[k]
                            dy_jk = new_yj - y[k]
                            if dx_jk * dx_jk + dy_jk * dy_jk < (r[j] + r[k]) ** 2 - 1e-9:
                                valid = False
                                break
                        
                        if valid:
                            # Check if this creates more space
                            x_test = x.copy()
                            y_test = y.copy()
                            x_test[i] = new_xi
                            y_test[i] = new_yi
                            x_test[j] = new_xj
                            y_test[j] = new_yj
                            
                            # Try to expand after move
                            r_test = r.copy()
                            for _ in range(15):
                                r_trial = r_test * 1.003
                                if _is_valid(x_test, y_test, r_trial, w, h):
                                    r_test = r_trial
                                else:
                                    break
                            
                            if np.sum(r_test) > np.sum(r) + 1e-6:
                                x[i] = new_xi
                                y[i] = new_yi
                                x[j] = new_xj
                                y[j] = new_yj
                                r = r_test
                                improved = True
                                break

    # Final uniform expansion - try scaling all radii
    for _ in range(10000):
        if time.time() - start_time > time_budget:
            break
        r_trial = r * 1.0002
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    # Second pass: try individual expansion again after uniform scaling
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
        if r_max > r[i] + 1e-7:
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
            if lo > r[i] + 1e-8:
                r[i] = lo

    # Third pass: final fine-tuning
    if time.time() - start_time < time_budget - 1:
        for _ in range(3):
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

    if _is_valid(x, y, r, w, h):
        return x, y, r
    return None, None, None


def _optimize_aspect(x, y, r, w0, h0, time_budget):
    """
    Optimize aspect ratio (width) by trying different widths.
    Uses a finer search around the current best with re-optimization.
    """
    best_w = w0
    best_val = np.sum(r)
    start_time = time.time()
    
    # Coarse search - wider range
    for w in np.linspace(0.35, 1.65, 40):
        if time.time() - start_time > time_budget:
            break
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
    
    # Fine search around best
    for w in np.linspace(max(0.3, best_w - 0.15), min(1.7, best_w + 0.15), 40):
        if time.time() - start_time > time_budget:
            break
        h = 2.0 - w
        if h < 0.3:
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
    
    # Very fine search
    for w in np.linspace(max(0.3, best_w - 0.04), min(1.7, best_w + 0.04), 25):
        if time.time() - start_time > time_budget:
            break
        h = 2.0 - w
        if h < 0.3:
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

    # Generate initial guesses using multiple strategies
    def get_initial_guesses():
        guesses = []
        # Focus on aspect ratios that are likely optimal
        aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 0.85, 1.15, 0.75, 1.25, 0.95, 1.05, 0.88, 1.12, 0.82, 1.18, 0.78, 1.22, 0.92, 1.08, 0.86, 1.14, 0.84, 1.16, 0.87, 1.13, 0.93, 1.07, 0.97, 1.03]
        
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.35:
                continue
            
            # Try hexagonal packing
            positions = _hexagonal_packing(w0, h0, N)
            if len(positions) >= N:
                x = np.array([p[0] for p in positions])
                y = np.array([p[1] for p in positions])
                r = np.array([p[2] for p in positions])
                guesses.append((x, y, r, w0, h0))
            
            # Try grid packing
            positions = _grid_packing(w0, h0, N)
            if len(positions) >= N:
                x = np.array([p[0] for p in positions])
                y = np.array([p[1] for p in positions])
                r = np.array([p[2] for p in positions])
                guesses.append((x, y, r, w0, h0))
            
            # Try corner-anchored packing
            # Place 4 circles in corners, then fill center
            r_corner = min(w0, h0) * 0.22
            x_c = np.array([r_corner, w0-r_corner, r_corner, w0-r_corner])
            y_c = np.array([r_corner, r_corner, h0-r_corner, h0-r_corner])
            r_c = np.full(4, r_corner)
            
            # Fill remaining with grid in center
            remaining = N - 4
            if remaining > 0:
                # Grid in the central region
                cx_min = 2*r_corner * 0.9
                cx_max = w0 - 2*r_corner * 0.9
                cy_min = 2*r_corner * 0.9
                cy_max = h0 - 2*r_corner * 0.9
                if cx_max > cx_min and cy_max > cy_min:
                    grid_positions = _grid_packing(cx_max - cx_min, cy_max - cy_min, remaining)
                    if len(grid_positions) >= remaining:
                        x_g = np.array([p[0] + cx_min for p in grid_positions[:remaining]])
                        y_g = np.array([p[1] + cy_min for p in grid_positions[:remaining]])
                        r_g = np.array([p[2] for p in grid_positions[:remaining]])
                        x = np.concatenate([x_c, x_g])
                        y = np.concatenate([y_c, y_g])
                        r = np.concatenate([r_c, r_g])
                        guesses.append((x, y, r, w0, h0))

        # Random starts with varied radii - more diverse
        rng = np.random.default_rng(42)
        for trial in range(80):
            w0 = rng.uniform(0.5, 1.5)
            h0 = 2.0 - w0
            # Better random initialization: cluster around potential good positions
            if trial % 3 == 0:
                # Hexagonal-like random
                r_est = 0.12
                x = []
                y = []
                for i in range(N):
                    row = i // 5
                    col = i % 5
                    base_x = (col + 0.5 + (row % 2) * 0.5) * (w0 / 5)
                    base_y = (row + 0.5) * (h0 / 5)
                    x.append(np.clip(base_x + rng.normal(0, 0.02), r_est, w0 - r_est))
                    y.append(np.clip(base_y + rng.normal(0, 0.02), r_est, h0 - r_est))
                x = np.array(x)
                y = np.array(y)
                r = np.full(N, r_est * 0.95)
            elif trial % 3 == 1:
                # Grid-like random
                r_est = 0.11
                x = []
                y = []
                for i in range(N):
                    row = i // 5
                    col = i % 5
                    base_x = (col + 0.5) * (w0 / 5)
                    base_y = (row + 0.5) * (h0 / 5)
                    x.append(np.clip(base_x + rng.normal(0, 0.015), r_est, w0 - r_est))
                    y.append(np.clip(base_y + rng.normal(0, 0.015), r_est, h0 - r_est))
                x = np.array(x)
                y = np.array(y)
                r = np.full(N, r_est * 0.95)
            else:
                # Pure random
                x = rng.uniform(0.05, w0 - 0.05, N)
                y = rng.uniform(0.05, h0 - 0.05, N)
                r_min = rng.uniform(0.04, 0.08)
                r_max = rng.uniform(0.10, 0.14)
                r = rng.uniform(r_min, r_max, N)
            guesses.append((x, y, r, w0, h0))

        return guesses

    guesses = get_initial_guesses()

    # Sort guesses by initial energy (lower is better)
    def guess_energy(g):
        x, y, r, w, h = g
        return _compute_energy(x, y, r, w, h)
    
    guesses.sort(key=guess_energy)

    # Process top guesses with more time, lower ones with less
    n_guesses = len(guesses)
    for i, (x0, y0, r0, w0, h0) in enumerate(guesses):
        if time.time() > deadline - 20:
            break

        # Allocate time based on rank: top guesses get more time
        rank_frac = i / max(1, n_guesses)
        time_alloc = 1.0 - 0.6 * rank_frac  # Top guess gets 1.0, bottom gets 0.4

        # Step 1: Dynamics simulation to resolve overlaps
        remaining = deadline - (time.time() - start_time)
        sim_time = min(5.0 * time_alloc, remaining * 0.2)
        x_sim, y_sim, r_sim = _dynamics_simulate(x0, y0, r0, w0, h0, sim_time)
        if x_sim is None:
            continue

        # Step 2: Optimize aspect ratio
        remaining = deadline - (time.time() - start_time)
        aspect_time = min(2.5 * time_alloc, remaining * 0.1)
        w_opt = _optimize_aspect(x_sim, y_sim, r_sim, w0, h0, aspect_time)
        if abs(w_opt - w0) > 1e-6:
            h_opt = 2.0 - w_opt
            scale_x = w_opt / w0
            scale_y = h_opt / h0
            x_sim = x_sim * scale_x
            y_sim = y_sim * scale_y
            r_sim = r_sim * min(scale_x, scale_y)
            w0 = w_opt
            h0 = h_opt
            
            # Re-run short dynamics to allow rearrangement after scaling
            sim_time_short = min(1.5 * time_alloc, remaining * 0.04)
            x_re, y_re, r_re = _dynamics_simulate(x_sim, y_sim, r_sim, w0, h0, sim_time_short)
            if x_re is not None:
                x_sim, y_sim, r_sim = x_re, y_re, r_re

        # Step 3: Expand radii
        remaining = deadline - (time.time() - start_time)
        expand_time = min(18.0 * time_alloc, remaining * 0.5)
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
    if best_result is not None and time.time() < deadline - 3:
        x_best, y_best, r_best, w_best, h_best = best_result
        final_time = max(3.0, deadline - (time.time() - start_time) - 3)
        x_final, y_final, r_final = _expand_radii(x_best, y_best, r_best, w_best, h_best, final_time)
        if x_final is not None:
            valid, val = check_validity(x_final, y_final, r_final, w_best, h_best)
            if valid and val > best_value:
                best_value = val
                best_result = (x_final.copy(), y_final.copy(), r_final.copy(), w_best, h_best)
        
        # Second polish: try aspect optimization again on the improved solution
        if time.time() < deadline - 2:
            x_b, y_b, r_b, w_b, h_b = best_result
            w_opt2 = _optimize_aspect(x_b, y_b, r_b, w_b, h_b, 1.5)
            if abs(w_opt2 - w_b) > 1e-6:
                h_opt2 = 2.0 - w_opt2
                scale_x = w_opt2 / w_b
                scale_y = h_opt2 / h_b
                x_b2 = x_b * scale_x
                y_b2 = y_b * scale_y
                r_b2 = r_b * min(scale_x, scale_y)
                
                # Re-run short dynamics to allow rearrangement
                sim_time_short = min(1.5, deadline - (time.time() - start_time) - 2)
                x_re, y_re, r_re = _dynamics_simulate(x_b2, y_b2, r_b2, w_opt2, h_opt2, sim_time_short)
                if x_re is not None:
                    x_b2, y_b2, r_b2 = x_re, y_re, r_re
                    # Expand radii after dynamics
                    expand_time = min(2.0, deadline - (time.time() - start_time) - 1.5)
                    x_e, y_e, r_e = _expand_radii(x_b2, y_b2, r_b2, w_opt2, h_opt2, expand_time)
                    if x_e is not None:
                        valid2, val2 = check_validity(x_e, y_e, r_e, w_opt2, h_opt2)
                        if valid2 and val2 > best_value:
                            best_value = val2
                            best_result = (x_e.copy(), y_e.copy(), r_e.copy(), w_opt2, h_opt2)
                        else:
                            valid2, val2 = check_validity(x_b2, y_b2, r_b2, w_opt2, h_opt2)
                            if valid2 and val2 > best_value:
                                best_value = val2
                                best_result = (x_b2.copy(), y_b2.copy(), r_b2.copy(), w_opt2, h_opt2)
                else:
                    valid2, val2 = check_validity(x_b2, y_b2, r_b2, w_opt2, h_opt2)
                    if valid2 and val2 > best_value:
                        best_value = val2
                        best_result = (x_b2.copy(), y_b2.copy(), r_b2.copy(), w_opt2, h_opt2)
        
        # Third polish: expand radii again after aspect optimization
        if time.time() < deadline - 1:
            x_b, y_b, r_b, w_b, h_b = best_result
            polish_time = max(1.5, deadline - (time.time() - start_time) - 0.5)
            x_p, y_p, r_p = _expand_radii(x_b, y_b, r_b, w_b, h_b, polish_time)
            if x_p is not None:
                valid_p, val_p = check_validity(x_p, y_p, r_p, w_b, h_b)
                if valid_p and val_p > best_value:
                    best_value = val_p
                    best_result = (x_p.copy(), y_p.copy(), r_p.copy(), w_b, h_b)

    if best_result is None:
        # Fallback: simple grid in a 1x1 square
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

    # Final safety shrink
    for _ in range(20):
        r_trial = r * (1 - 1e-8)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
