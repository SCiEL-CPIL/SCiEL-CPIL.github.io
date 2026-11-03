"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
import time
import random
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _max_radius_for_positions(pos):
    """
    Vectorized: given positions (N,2), compute the maximum radius each circle
    can have such that all circles are disjoint and inside the unit square.
    r_i = min(boundary[i], min_{j != i} dist(i,j)/2)
    """
    dists = cdist(pos, pos)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists, axis=1) / 2.0
    boundary = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                          np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
    radii = np.minimum(boundary, min_pair)
    return np.maximum(radii, 0.0)


def _objective(pos):
    """Sum of max radii for given positions."""
    return float(np.sum(_max_radius_for_positions(pos)))


def _objective_single_move(pos, i, new_x, new_y):
    """
    Fast objective when only circle i moves to (new_x, new_y).
    Avoids full cdist; only computes distances from i to all others.
    """
    n = len(pos)
    # Boundary radius for circle i
    bx = min(new_x, 1.0 - new_x)
    by = min(new_y, 1.0 - new_y)
    r_i = min(bx, by)

    # Distances from i to all others
    dx = pos[:, 0] - new_x
    dy = pos[:, 1] - new_y
    dists_i = np.sqrt(dx * dx + dy * dy)
    dists_i[i] = np.inf
    r_i = min(r_i, np.min(dists_i) / 2.0)
    r_i = max(r_i, 0.0)

    # For each other circle j, recompute its radius:
    # r_j = min(boundary_j, min_{k != j} dist(j,k)/2)
    # The only change is dist(j,i) which changed.
    # r_j_new = min(boundary_j, min(old_min_except_i, dist(j,i)/2))
    # We need old radii and old min-pair for each j.
    # Simpler: recompute all radii but using the new position for i.
    # Actually for correctness we need full recompute since moving i changes
    # the constraint for all j. But we can do it faster than cdist.
    
    # Recompute all radii with updated position
    pos_new = pos.copy()
    pos_new[i, 0] = new_x
    pos_new[i, 1] = new_y
    
    # Vectorized distance computation (faster than cdist for this use)
    diff = pos_new[:, np.newaxis, :] - pos_new[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists, axis=1) / 2.0
    boundary = np.minimum(np.minimum(pos_new[:, 0], 1.0 - pos_new[:, 0]),
                          np.minimum(pos_new[:, 1], 1.0 - pos_new[:, 1]))
    radii = np.minimum(boundary, min_pair)
    radii = np.maximum(radii, 0.0)
    return float(np.sum(radii))


def _smooth_objective(v, n, beta=40.0):
    """
    Smooth objective for gradient-based polish.
    v = [x_0..x_{n-1}, y_0..y_{n-1}, r_0..r_{n-1}]
    Minimize: -sum(r) + penalty * (smooth wall violations + smooth pairwise violations)
    """
    x = v[0:n]
    y = v[n:2 * n]
    r = v[2 * n:3 * n]

    # Wall violations: r_i - min(x_i, 1-x_i, y_i, 1-y_i)
    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    wall_pen = np.sum(np.logaddexp(0.0, beta * wall_viol)) / beta

    # Pairwise violations: r_i + r_j - dist_ij
    total_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            viol = r[i] + r[j] - dist
            total_pen += np.logaddexp(0.0, beta * viol) / beta

    return -np.sum(r) + 50.0 * (wall_pen + total_pen)


def _smooth_gradient(v, n, beta=40.0):
    """
    Vectorized analytic gradient of _smooth_objective.
    """
    x = v[0:n]
    y = v[n:2 * n]
    r = v[2 * n:3 * n]

    gx = np.zeros(n)
    gy = np.zeros(n)
    gr = np.zeros(n)

    # Wall terms: for each i, wall_viol_i = r_i - min(x_i, 1-x_i, y_i, 1-y_i)
    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    sig_wall = 1.0 / (1.0 + np.exp(-beta * wall_viol))
    sig_wall = np.where(sig_wall > 1e-12, sig_wall, 0.0)

    # Determine which wall is binding for each circle
    # dists to 4 walls
    d0 = x          # left
    d1 = 1.0 - x    # right
    d2 = y          # bottom
    d3 = 1.0 - y    # top
    wall_dists = np.stack([d0, d1, d2, d3], axis=1)  # (n, 4)
    min_idx = np.argmin(wall_dists, axis=1)  # (n,)

    factor = 50.0 * sig_wall  # (n,)

    # Gradient contribution to r
    gr += factor

    # Gradient contribution to x, y based on which wall is min
    # For min_idx == 0 (left): d(wall_viol)/dx = -1, so gx -= factor
    # For min_idx == 1 (right): d(wall_viol)/dx = +1, so gx += factor
    # For min_idx == 2 (bottom): d(wall_viol)/dy = -1, so gy -= factor
    # For min_idx == 3 (top): d(wall_viol)/dy = +1, so gy += factor
    left_mask = (min_idx == 0)
    right_mask = (min_idx == 1)
    bottom_mask = (min_idx == 2)
    top_mask = (min_idx == 3)

    gx[left_mask] -= factor[left_mask]
    gx[right_mask] += factor[right_mask]
    gy[bottom_mask] -= factor[bottom_mask]
    gy[top_mask] += factor[top_mask]

    # Pairwise terms: for each pair (i,j), viol = r_i + r_j - dist_ij
    # s_ij = sigmoid(beta * viol_ij)
    # grad_x_i += 50*s*dx/dist, grad_x_j -= 50*s*dx/dist, etc.
    # Vectorized over all pairs
    dx_mat = x[:, np.newaxis] - x[np.newaxis, :]  # (n, n)
    dy_mat = y[:, np.newaxis] - y[np.newaxis, :]
    dist_mat = np.sqrt(dx_mat * dx_mat + dy_mat * dy_mat)
    np.fill_diagonal(dist_mat, 1.0)  # avoid div by zero (masked out)

    viol_mat = r[:, np.newaxis] + r[np.newaxis, :] - dist_mat
    np.fill_diagonal(viol_mat, -np.inf)  # mask diagonal

    sig_mat = 1.0 / (1.0 + np.exp(-beta * viol_mat))
    sig_mat = np.where((sig_mat > 1e-12) & (np.arange(n)[:, np.newaxis] != np.arange(n)[np.newaxis, :]), sig_mat, 0.0)

    # factor_mat = 50 * sig_mat
    factor_mat = 50.0 * sig_mat

    # grad_x_i = sum_j factor_mat[i,j] * dx_mat[i,j] / dist_mat[i,j]
    grad_x_pair = np.sum(factor_mat * dx_mat / dist_mat, axis=1)
    grad_y_pair = np.sum(factor_mat * dy_mat / dist_mat, axis=1)
    grad_r_pair = np.sum(factor_mat, axis=1)

    gx += grad_x_pair
    gy += grad_y_pair
    gr += grad_r_pair

    # Objective: -sum(r)
    gr -= 1.0

    grad = np.concatenate([gx, gy, gr])
    return grad


def _is_valid(pos, eps=1e-9):
    """Check pairwise non-overlap and boundary containment with computed radii."""
    radii = _max_radius_for_positions(pos)
    dists = cdist(pos, pos)
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            if dists[i, j] < radii[i] + radii[j] - eps:
                return False
        if (pos[i, 0] < radii[i] - eps or pos[i, 0] > 1.0 - radii[i] + eps or
                pos[i, 1] < radii[i] - eps or pos[i, 1] > 1.0 - radii[i] + eps):
            return False
    return True


def _initial_grid():
    side = int(np.ceil(np.sqrt(N)))
    pos = []
    for i in range(side):
        for j in range(side):
            if len(pos) >= N:
                break
            pos.append([(i + 0.5) / side, (j + 0.5) / side])
    return np.array(pos[:N])


def _initial_random(seed=0):
    rng = np.random.RandomState(seed)
    pos = []
    attempts = 0
    while len(pos) < N and attempts < 5000:
        attempts += 1
        x, y = rng.random(), rng.random()
        if all(np.hypot(x - px, y - py) > 0.06 for px, py in pos):
            pos.append([x, y])
    while len(pos) < N:
        pos.append([rng.random(), rng.random()])
    return np.array(pos)


def _simulated_annealing(pos, t_start=0.5, t_end=0.0005, n_iter=40000, seed=42, deadline=None):
    """
    Simulated annealing to maximize sum of radii.
    Uses vectorized _max_radius_for_positions for speed.
    """
    rng = np.random.RandomState(seed)
    current_pos = pos.copy()
    current_obj = _objective(current_pos)

    best_pos = current_pos.copy()
    best_obj = current_obj

    n = len(current_pos)

    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break

        t = t_start * (t_end / t_start) ** (it / n_iter)

        i = rng.randint(n)
        step_size = 0.04 * t + 0.0008
        dx = rng.uniform(-step_size, step_size)
        dy = rng.uniform(-step_size, step_size)

        new_pos = current_pos.copy()
        new_pos[i, 0] = np.clip(current_pos[i, 0] + dx, 0.0, 1.0)
        new_pos[i, 1] = np.clip(current_pos[i, 1] + dy, 0.0, 1.0)

        new_obj = _objective(new_pos)
        delta = new_obj - current_obj

        if delta > 0 or rng.random() < np.exp(delta / t):
            current_pos = new_pos
            current_obj = new_obj
            if current_obj > best_obj:
                best_pos = current_pos.copy()
                best_obj = current_obj

    return best_pos, best_obj


def _local_search(pos, n_iter=4000, step_size=0.008, seed=42, deadline=None):
    """
    Local search: iteratively nudge each circle to improve the objective.
    Uses fast single-move objective evaluation.
    """
    rng = np.random.RandomState(seed)
    current_pos = pos.copy()
    current_obj = _objective(current_pos)

    n = len(current_pos)
    improved = True
    it = 0
    while improved and it < n_iter:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        it += 1
        order = rng.permutation(n)
        for i in order:
            for _ in range(15):
                dx = rng.uniform(-step_size, step_size)
                dy = rng.uniform(-step_size, step_size)
                nx = np.clip(current_pos[i, 0] + dx, 0.0, 1.0)
                ny = np.clip(current_pos[i, 1] + dy, 0.0, 1.0)

                new_obj = _objective_single_move(current_pos, i, nx, ny)
                if new_obj > current_obj + 1e-13:
                    current_pos[i, 0] = nx
                    current_pos[i, 1] = ny
                    current_obj = new_obj
                    improved = True
                    break

    return current_pos, current_obj


def _best_fit_local_search(pos, n_iter=8000, seed=42, deadline=None):
    """
    Dense grid-based best-fit local search: for each circle, search a fine grid
    around its current position to find the best new location.
    Uses fast single-move objective.
    """
    rng = np.random.RandomState(seed)
    current_pos = pos.copy()
    current_obj = _objective(current_pos)
    n = len(current_pos)

    base_step = 0.015
    grid_res = 5

    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break

        improved_any = False
        order = rng.permutation(n)
        
        step_scale = max(0.2, 1.0 - it / (n_iter * 0.7))
        step = base_step * step_scale

        for i in order:
            if deadline is not None and time.time() > deadline:
                break

            x_i, y_i = current_pos[i]
            
            best_new_x = None
            best_new_y = None
            best_new_obj = current_obj

            for di in range(-grid_res, grid_res + 1):
                for dj in range(-grid_res, grid_res + 1):
                    if di == 0 and dj == 0:
                        continue
                    cx = x_i + di * step
                    cy = y_i + dj * step
                    if cx < 0.0 or cx > 1.0 or cy < 0.0 or cy > 1.0:
                        continue
                    new_obj = _objective_single_move(current_pos, i, cx, cy)
                    if new_obj > best_new_obj + 1e-14:
                        best_new_obj = new_obj
                        best_new_x = cx
                        best_new_y = cy

            for _ in range(8):
                angle = rng.uniform(0, 2 * np.pi)
                mag = step * rng.uniform(0.5, 1.5)
                cx = np.clip(x_i + mag * np.cos(angle), 0.0, 1.0)
                cy = np.clip(y_i + mag * np.sin(angle), 0.0, 1.0)
                new_obj = _objective_single_move(current_pos, i, cx, cy)
                if new_obj > best_new_obj + 1e-14:
                    best_new_obj = new_obj
                    best_new_x = cx
                    best_new_y = cy

            if best_new_x is not None:
                current_pos[i, 0] = best_new_x
                current_pos[i, 1] = best_new_y
                current_obj = best_new_obj
                improved_any = True

        if not improved_any:
            for _ in range(40):
                if deadline is not None and time.time() > deadline:
                    break
                i = rng.randint(n)
                x_i, y_i = current_pos[i]
                dx = rng.uniform(-step * 0.5, step * 0.5)
                dy = rng.uniform(-step * 0.5, step * 0.5)
                nx = np.clip(x_i + dx, 0.0, 1.0)
                ny = np.clip(y_i + dy, 0.0, 1.0)
                new_obj = _objective_single_move(current_pos, i, nx, ny)
                if new_obj > current_obj + 1e-14:
                    current_pos[i, 0] = nx
                    current_pos[i, 1] = ny
                    current_obj = new_obj
                    improved_any = True
            if not improved_any:
                base_step *= 0.6
                if base_step < 0.0003:
                    break

    return current_pos, current_obj


def _high_precision_polish(pos, deadline):
    """
    High-precision final polish: run gradient-based optimization with progressive
    beta values to smooth the landscape initially and sharpen it later.
    """
    n = len(pos)
    current_pos = pos.copy()
    r_init = _max_radius_for_positions(current_pos)
    best_pos = current_pos.copy()
    best_r = r_init.copy()
    best_obj = float(np.sum(r_init))
    
    betas = [20.0, 40.0, 80.0, 150.0, 250.0, 400.0, 600.0, 800.0]
    
    for beta in betas:
        if time.time() > deadline - 2:
            break
            
        r_current = _max_radius_for_positions(current_pos)
        v0 = np.concatenate([current_pos[:, 0], current_pos[:, 1], r_current])

        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0))
        for i in range(n):
            bounds.append((0.0, 1.0))
        for i in range(n):
            bounds.append((0.0, 0.5))

        try:
            maxiter = 1500 if beta >= 150 else 800
            res = minimize(
                _smooth_objective, v0, args=(n, beta),
                jac=_smooth_gradient, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-18, 'gtol': 1e-15, 'maxls': 100}
            )
        except Exception:
            continue

        if time.time() > deadline - 2:
            break

        v_opt = res.x
        x_opt = v_opt[0:n]
        y_opt = v_opt[n:2 * n]
        pos_opt = np.column_stack([x_opt, y_opt])
        
        pos_opt = np.clip(pos_opt, 0.0, 1.0)
        
        r_exact = _max_radius_for_positions(pos_opt)
        obj_opt = float(np.sum(r_exact))

        if _is_valid(pos_opt, eps=1e-9) and obj_opt > best_obj:
            best_pos = pos_opt
            best_r = r_exact
            best_obj = obj_opt
            current_pos = pos_opt.copy()
        elif _is_valid(pos_opt, eps=1e-9) and obj_opt > best_obj - 1e-10:
            current_pos = pos_opt.copy()

    # Final micro-search with fast objective
    if time.time() < deadline - 0.3:
        rng = np.random.RandomState(42)
        for _ in range(1000):
            if time.time() > deadline - 0.1:
                break
            i = rng.randint(n)
            dx = rng.uniform(-0.0003, 0.0003)
            dy = rng.uniform(-0.0003, 0.0003)
            nx = np.clip(best_pos[i, 0] + dx, 0.0, 1.0)
            ny = np.clip(best_pos[i, 1] + dy, 0.0, 1.0)
            new_obj = _objective_single_move(best_pos, i, nx, ny)
            if new_obj > best_obj + 1e-12:
                best_pos[i, 0] = nx
                best_pos[i, 1] = ny
                best_obj = new_obj

    return best_pos, best_r, best_obj


def _gradient_polish(pos, deadline, beta=40.0):
    """
    Final polish: use L-BFGS-B to optimize [x, y, r] jointly with smooth objective.
    Returns improved (pos, radii) if valid and better.
    """
    n = len(pos)
    r_init = _max_radius_for_positions(pos)
    v0 = np.concatenate([pos[:, 0], pos[:, 1], r_init])

    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((0.0, 0.5))

    try:
        res = minimize(
            _smooth_objective, v0, args=(n, beta),
            jac=_smooth_gradient, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 800, 'ftol': 1e-17, 'gtol': 1e-14, 'maxls': 100}
        )
    except Exception:
        return pos, r_init

    if time.time() > deadline:
        return pos, r_init

    v_opt = res.x
    x_opt = v_opt[0:n]
    y_opt = v_opt[n:2 * n]

    pos_opt = np.column_stack([x_opt, y_opt])
    pos_opt = np.clip(pos_opt, 0.0, 1.0)
    r_exact = _max_radius_for_positions(pos_opt)
    obj_opt = float(np.sum(r_exact))

    if _is_valid(pos_opt, eps=1e-9):
        return pos_opt, r_exact, obj_opt

    return pos, r_init, float(np.sum(r_init))


def _bubble_relaxation(pos, deadline, n_iter=2000, dt=0.001):
    """
    Physics-based bubble relaxation: circles are modeled as pressurized gas bubbles.
    Internal pressure is proportional to r_i^2, creating an outward force.
    Repulsive contact forces between circles and boundary constraints are balanced.
    The system naturally maximizes sum of radii through mechanical equilibrium.
    """
    n = len(pos)
    current_pos = pos.copy()
    r_current = _max_radius_for_positions(current_pos)
    best_pos = current_pos.copy()
    best_obj = float(np.sum(r_current))
    
    # Velocity arrays for momentum
    vx = np.zeros(n)
    vy = np.zeros(n)
    
    # Damping factor
    damping = 0.95
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
        
        # Compute forces
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Boundary repulsion: push circles away from walls
        for i in range(n):
            x, y = current_pos[i]
            r = r_current[i]
            
            # Left wall
            if x < r:
                fx[i] += (r - x) * 100.0
            # Right wall
            if x > 1.0 - r:
                fx[i] -= (x - (1.0 - r)) * 100.0
            # Bottom wall
            if y < r:
                fy[i] += (r - y) * 100.0
            # Top wall
            if y > 1.0 - r:
                fy[i] -= (y - (1.0 - r)) * 100.0
        
        # Pairwise repulsion: circles push each other apart
        for i in range(n):
            for j in range(i + 1, n):
                dx = current_pos[j, 0] - current_pos[i, 0]
                dy = current_pos[j, 1] - current_pos[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                
                if dist < 1e-10:
                    # Random small perturbation to break symmetry
                    dx = np.random.uniform(-0.001, 0.001)
                    dy = np.random.uniform(-0.001, 0.001)
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < 1e-10:
                        continue
                
                # Overlap amount
                overlap = r_current[i] + r_current[j] - dist
                
                if overlap > 0:
                    # Repulsive force proportional to overlap
                    force_mag = overlap * 50.0
                    fx[i] -= force_mag * dx / dist
                    fy[i] -= force_mag * dy / dist
                    fx[j] += force_mag * dx / dist
                    fy[j] += force_mag * dy / dist
        
        # Update velocities with damping
        vx = vx * damping + fx * dt
        vy = vy * damping + fy * dt
        
        # Update positions
        current_pos[:, 0] += vx * dt
        current_pos[:, 1] += vy * dt
        
        # Clamp to unit square
        current_pos[:, 0] = np.clip(current_pos[:, 0], 0.0, 1.0)
        current_pos[:, 1] = np.clip(current_pos[:, 1], 0.0, 1.0)
        
        # Recompute radii and check improvement
        r_new = _max_radius_for_positions(current_pos)
        obj_new = float(np.sum(r_new))
        
        if obj_new > best_obj + 1e-12:
            best_pos = current_pos.copy()
            best_obj = obj_new
            r_current = r_new
    
    return best_pos, best_obj


def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill

    # Diverse initializations
    initializations = [_initial_grid()]
    for s in range(6):
        initializations.append(_initial_random(seed=s * 997 + 13))

    best_pos = None
    best_obj = -np.inf

    # Phase 1: Simulated annealing + best-fit local search from each init
    # Budget: ~40s for SA phase
    for init_idx, init_pos in enumerate(initializations):
        if time.time() > deadline - 40:
            break

        sa_pos, sa_obj = _simulated_annealing(
            init_pos, t_start=0.4, t_end=0.0003,
            n_iter=25000, seed=init_idx * 7 + 1,
            deadline=deadline - 38
        )

        if sa_obj > best_obj:
            best_pos = sa_pos
            best_obj = sa_obj

        if time.time() > deadline - 38:
            break

        # Best-fit local search (targeted moves)
        bf_pos, bf_obj = _best_fit_local_search(
            sa_pos, n_iter=4000,
            seed=init_idx * 13 + 2,
            deadline=deadline - 36
        )

        if bf_obj > best_obj:
            best_pos = bf_pos
            best_obj = bf_obj

        # Additional standard local search pass
        if time.time() > deadline - 34:
            break
        ls_pos, ls_obj = _local_search(
            bf_pos, n_iter=2000, step_size=0.004,
            seed=init_idx * 17 + 3,
            deadline=deadline - 32
        )

        if ls_obj > best_obj:
            best_pos = ls_pos
            best_obj = ls_obj

    if best_pos is None:
        pos = _initial_grid()
        r = _max_radius_for_positions(pos)
        return np.column_stack([pos, r])

    # Phase 2: Best-fit local search on best solution
    if time.time() < deadline - 30:
        bf_final, bf_final_obj = _best_fit_local_search(
            best_pos, n_iter=8000,
            seed=777, deadline=deadline - 28
        )
        if bf_final_obj > best_obj:
            best_pos = bf_final
            best_obj = bf_final_obj

    # Phase 3: Standard local search with finer steps
    if time.time() < deadline - 24:
        final_pos, final_obj = _local_search(
            best_pos, n_iter=4000, step_size=0.002,
            seed=999, deadline=deadline - 22
        )
        if final_obj > best_obj:
            best_pos = final_pos
            best_obj = final_obj

    # Phase 4: Bubble relaxation
    if time.time() < deadline - 22:
        bubble_pos, bubble_obj = _bubble_relaxation(
            best_pos, deadline=deadline - 20,
            n_iter=2000, dt=0.001
        )
        if bubble_obj > best_obj:
            best_pos = bubble_pos
            best_obj = bubble_obj

    # Phase 5: Gradient-based polish (smooth surrogate) - multiple passes
    if time.time() < deadline - 18:
        polished = _gradient_polish(best_pos, deadline - 16, beta=40.0)
        if len(polished) == 3:
            p_pol, r_pol, obj_pol = polished
            if obj_pol > best_obj:
                best_pos = p_pol
                best_obj = obj_pol

    # Phase 6: High-precision final polish (progressive beta) - main refinement
    if time.time() < deadline - 10:
        hp_pos, hp_r, hp_obj = _high_precision_polish(best_pos, deadline - 4)
        if hp_obj > best_obj:
            best_pos = hp_pos
            best_obj = hp_obj

    # Phase 7: Another gradient polish after high-precision
    if time.time() < deadline - 6:
        polished2 = _gradient_polish(best_pos, deadline - 5, beta=80.0)
        if len(polished2) == 3:
            p_pol2, r_pol2, obj_pol2 = polished2
            if obj_pol2 > best_obj:
                best_pos = p_pol2
                best_obj = obj_pol2

    # Phase 8: Best-fit pass after gradient polish
    if time.time() < deadline - 4:
        bf_after_grad, bf_ag_obj = _best_fit_local_search(
            best_pos, n_iter=4000,
            seed=555, deadline=deadline - 3
        )
        if bf_ag_obj > best_obj:
            best_pos = bf_after_grad
            best_obj = bf_ag_obj

    # Phase 9: Final fine-tuning local search with very small steps
    if time.time() < deadline - 2:
        final_pos, final_obj = _local_search(
            best_pos, n_iter=2000, step_size=0.0006,
            seed=12345, deadline=deadline - 1
        )
        if final_obj > best_obj:
            best_pos = final_pos
            best_obj = final_obj

    # Phase 10: Ultra-fine polish
    if time.time() < deadline - 0.3:
        rng = np.random.RandomState(99999)
        for _ in range(400):
            if time.time() > deadline - 0.05:
                break
            i = rng.randint(len(best_pos))
            dx = rng.uniform(-0.0002, 0.0002)
            dy = rng.uniform(-0.0002, 0.0002)
            nx = np.clip(best_pos[i, 0] + dx, 0.0, 1.0)
            ny = np.clip(best_pos[i, 1] + dy, 0.0, 1.0)
            new_obj = _objective_single_move(best_pos, i, nx, ny)
            if new_obj > best_obj + 1e-13:
                best_pos[i, 0] = nx
                best_pos[i, 1] = ny
                best_obj = new_obj

    # Build result
    r_final = _max_radius_for_positions(best_pos)
    result = np.column_stack([best_pos, r_final])
    return result

# EVOLVE-BLOCK-END
