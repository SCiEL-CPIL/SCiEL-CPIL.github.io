"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

# Precompute pair indices
_II, _JJ = np.triu_indices(N, k=1)
_NPAIRS = len(_II)


def _soft_max(v, beta):
    """Smooth approximation of max(0, v) via softplus."""
    v = np.asarray(v, dtype=float)
    return np.where(v > 20.0 / beta, v, np.log1p(np.exp(beta * v)) / beta)


def _vectorized_check(pos, radii, tol=1e-10):
    """
    Fully vectorized validity check. Returns (is_valid, min_margin).
    min_margin is the minimum of all constraint margins (positive = valid).
    """
    # Boundary margins
    m_left = pos[:, 0] - radii
    m_right = 1.0 - pos[:, 0] - radii
    m_bottom = pos[:, 1] - radii
    m_top = 1.0 - pos[:, 1] - radii
    boundary_min = min(m_left.min(), m_right.min(), m_bottom.min(), m_top.min())

    # Overlap margins: dist - (ri + rj)
    dx = pos[_II, 0] - pos[_JJ, 0]
    dy = pos[_II, 1] - pos[_JJ, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    overlap_margin = dist - (radii[_II] + radii[_JJ])
    overlap_min = overlap_margin.min()

    overall_min = min(boundary_min, overlap_min)
    return overall_min > -tol, overall_min


def _penalty_function(x, penalty_weight=200.0, beta=80.0):
    """
    Objective: minimize -sum(radii) + penalty * smooth_violations.
    Uses softplus for smooth max(0, v).
    """
    pos = x[:2 * N].reshape(N, 2)
    radii = np.maximum(x[2 * N:], 1e-7)

    # Boundary violations
    bv_left = radii - pos[:, 0]
    bv_right = pos[:, 0] + radii - 1.0
    bv_bottom = radii - pos[:, 1]
    bv_top = pos[:, 1] + radii - 1.0

    boundary_pen = (np.sum(_soft_max(bv_left, beta) ** 2) +
                    np.sum(_soft_max(bv_right, beta) ** 2) +
                    np.sum(_soft_max(bv_bottom, beta) ** 2) +
                    np.sum(_soft_max(bv_top, beta) ** 2))

    # Overlap violations in distance space
    dx = pos[_II, 0] - pos[_JJ, 0]
    dy = pos[_II, 1] - pos[_JJ, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    min_dist = radii[_II] + radii[_JJ]
    viol = min_dist - dist
    overlap_pen = np.sum(_soft_max(viol, beta) ** 2)

    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)


def _post_process(pos, radii):
    """
    Ensure strict validity by minimally adjusting radii/positions.
    Vectorized overlap check for speed.
    """
    pos = pos.copy()
    radii = radii.copy()

    for _ in range(300):
        changed = False

        # Boundary fix
        max_r_bound = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                                 np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
        over = radii > max_r_bound
        if np.any(over):
            radii[over] = max_r_bound[over] * (1.0 - 1e-10)
            changed = True

        # Overlap fix (vectorized detection, then fix)
        dx = pos[_II, 0] - pos[_JJ, 0]
        dy = pos[_II, 1] - pos[_JJ, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        total_r = radii[_II] + radii[_JJ]
        overlap = dist < total_r
        if np.any(overlap):
            for idx in np.where(overlap)[0]:
                i, j = _II[idx], _JJ[idx]
                d = np.sqrt((pos[i, 0] - pos[j, 0])**2 + (pos[i, 1] - pos[j, 1])**2)
                tr = radii[i] + radii[j]
                if tr > 1e-15:
                    scale = d / tr * (1.0 - 1e-10)
                    radii[i] *= scale
                    radii[j] *= scale
            changed = True

        if not changed:
            break

    # Final safety shrink
    valid, margin = _vectorized_check(pos, radii, tol=1e-10)
    if not valid:
        for _ in range(50):
            radii *= (1.0 - 1e-7)
            valid, margin = _vectorized_check(pos, radii, tol=1e-10)
            if valid:
                break

    return pos, radii


def _best_fit_init(seed=0):
    """
    Greedy Best-Fit initialization.
    """
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)

    pos[0] = [0.5, 0.5]
    radii[0] = 0.25

    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False

        for _ in range(30000):
            x = rng.uniform(0.03, 0.97)
            y = rng.uniform(0.03, 0.97)

            max_r_wall = min(x, 1.0 - x, y, 1.0 - y)
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r

            if max_r_circ > best_r:
                best_r = max_r_circ
                best_x = x
                best_y = y
                found = True

        if found and best_r > 0.005:
            pos[i] = [best_x, best_y]
            radii[i] = best_r * 0.97
        else:
            # Fallback: find any valid spot
            for _ in range(2000):
                x = rng.uniform(0.05, 0.95)
                y = rng.uniform(0.05, 0.95)
                min_dist = min(x, 1 - x, y, 1 - y)
                for j in range(i):
                    d = np.sqrt((x - pos[j, 0]) ** 2 + (y - pos[j, 1]) ** 2)
                    min_dist = min(min_dist, d - radii[j])
                if min_dist > 0.003:
                    pos[i] = [x, y]
                    radii[i] = min_dist * 0.93
                    break
            else:
                pos[i] = [0.5, 0.5]
                radii[i] = 0.001

    return pos, radii


def _hex_init():
    """Hexagonal lattice initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)

    r = 0.115
    dx = r * np.sqrt(3)
    dy = r * 1.5

    k = 0
    for row in range(6):
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        y = row * dy + r

        for col in range(6):
            if k >= N:
                break
            x = start_x + col * dx
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
        if k >= N:
            break

    rng = np.random.RandomState(456)
    while k < N:
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        for _ in range(8000):
            x = rng.uniform(0.03, 0.97)
            y = rng.uniform(0.03, 0.97)
            max_r_wall = min(x, 1.0 - x, y, 1.0 - y)
            max_r_circ = max_r_wall
            for j in range(k):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            if max_r_circ > best_r:
                best_r = max_r_circ
                best_x = x
                best_y = y
        if best_r > 0.005:
            pos[k] = [best_x, best_y]
            radii[k] = best_r * 0.97
        else:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
        k += 1

    return pos, radii


def _grid_init():
    """Grid-based initialization with varying radii."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    n = 5
    r = 0.095
    k = 0
    for i in range(n):
        for j in range(n):
            if k >= N:
                break
            pos[k] = [(i + 0.5) / n, (j + 0.5) / n]
            radii[k] = r
            k += 1
    return pos, radii


def _force_directed_relax(pos, radii, deadline, max_iter=500):
    """
    Force-directed relaxation to separate overlapping circles.
    This is a heuristic step to improve the starting configuration for gradient methods.
    """
    pos = pos.copy()
    radii = radii.copy()
    step_size = 0.01
    
    for _ in range(max_iter):
        if time.time() > deadline - 10:
            break
        
        forces = np.zeros((N, 2))
        
        # Wall forces
        # Left
        for i in range(N):
            if pos[i, 0] < radii[i]:
                forces[i, 0] += (radii[i] - pos[i, 0])
            if pos[i, 0] > 1.0 - radii[i]:
                forces[i, 0] -= (pos[i, 0] - (1.0 - radii[i]))
            if pos[i, 1] < radii[i]:
                forces[i, 1] += (radii[i] - pos[i, 1])
            if pos[i, 1] > 1.0 - radii[i]:
                forces[i, 1] -= (pos[i, 1] - (1.0 - radii[i]))
                
        # Pairwise repulsion
        dx = pos[_II, 0] - pos[_JJ, 0]
        dy = pos[_II, 1] - pos[_JJ, 1]
        dist = np.sqrt(dx*dx + dy*dy)
        dist = np.maximum(dist, 1e-8)
        
        min_dist = radii[_II] + radii[_JJ]
        overlap = min_dist - dist
        active = overlap > 0
        
        if np.any(active):
            # Force is proportional to overlap, direction is away
            # We apply equal and opposite forces
            f_mag = overlap[active]
            dir_x = dx[active] / dist[active]
            dir_y = dy[active] / dist[active]
            
            # Scatter update is tricky with indices, so we use a simple loop for active pairs
            # Or vectorized accumulation
            # For efficiency in Python, looping over active pairs (usually few if relaxed) is okay, 
            # but let's try vectorized accumulation if possible. 
            # Since _II and _JJ are fixed, we can use np.add.at
            
            np.add.at(forces, _II[active], -np.column_stack([dir_x, dir_y]) * f_mag[:, None])
            np.add.at(forces, _JJ[active], np.column_stack([dir_x, dir_y]) * f_mag[:, None])

        # Update positions
        pos += forces * step_size
        
        # Clamp to boundaries (roughly)
        pos[:, 0] = np.clip(pos[:, 0], 0.01, 0.99)
        pos[:, 1] = np.clip(pos[:, 1], 0.01, 0.99)
        
        # Decay step size
        step_size *= 0.99

    return pos, radii

def _local_search(pos, radii, deadline, max_iter=10000, step_scale=1.0):
    """
    Fast local search with vectorized validation and simulated annealing.
    """
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    current_pos = pos.copy()
    current_radii = radii.copy()

    rng = np.random.RandomState(777)
    temp = 0.01
    cooling = 0.999
    
    # Precompute for speed
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break

        new_pos = current_pos.copy()
        new_radii = current_radii.copy()

        n_perturb = rng.randint(1, 6)
        indices = rng.choice(N, n_perturb, replace=False)
        for idx in indices:
            # Larger steps initially
            scale = step_scale * (1.0 + 0.5 * np.exp(-it/100.0))
            new_pos[idx, 0] += rng.uniform(-0.02, 0.02) * scale
            new_pos[idx, 1] += rng.uniform(-0.02, 0.02) * scale
            new_radii[idx] *= (1 + rng.uniform(-0.05, 0.05) * scale)
            new_radii[idx] = max(new_radii[idx], 1e-7)

        # Fast vectorized check
        valid, margin = _vectorized_check(new_pos, new_radii, tol=-1e-9)
        if valid:
            new_score = np.sum(new_radii)
            if new_score > best_score:
                best_score = new_score
                best_pos = new_pos.copy()
                best_radii = new_radii.copy()
            delta = new_score - best_score
            if delta < 0 and rng.uniform() < np.exp(delta / max(temp, 1e-12)):
                current_pos = new_pos.copy()
                current_radii = new_radii.copy()
            elif new_score >= best_score:
                current_pos = new_pos.copy()
                current_radii = new_radii.copy()

        temp *= cooling

    return best_pos, best_radii


def _polish(pos, radii, deadline):
    """
    High-precision polish: SLSQP with very tight tolerances.
    """
    x0 = np.concatenate([pos.flatten(), radii])

    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.0, 0.5))

    def constraint_boundary(x):
        p = x[:2 * N].reshape(N, 2)
        r = x[2 * N:]
        c1 = p[:, 0] - r
        c2 = 1.0 - p[:, 0] - r
        c3 = p[:, 1] - r
        c4 = 1.0 - p[:, 1] - r
        return np.concatenate([c1, c2, c3, c4])

    def constraint_overlap(x):
        p = x[:2 * N].reshape(N, 2)
        r = x[2 * N:]
        dx = p[_II, 0] - p[_JJ, 0]
        dy = p[_II, 1] - p[_JJ, 1]
        dist_sq = dx * dx + dy * dy
        min_dist = r[_II] + r[_JJ]
        return dist_sq - min_dist * min_dist

    def objective(x):
        return -np.sum(x[2 * N:])

    constraints = [
        {'type': 'ineq', 'fun': constraint_boundary},
        {'type': 'ineq', 'fun': constraint_overlap}
    ]

    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-16, 'disp': False})
        if res.success or (res.fun < objective(x0)):
            pos_curr = res.x[:2 * N].reshape(N, 2)
            rad_curr = np.maximum(res.x[2 * N:], 1e-7)
            pos_pp, rad_pp = _post_process(pos_curr, rad_curr)
            valid, _ = _vectorized_check(pos_pp, rad_pp)
            if valid:
                return pos_pp, rad_pp
    except Exception:
        pass

    return pos, radii


def _optimize(pos, radii, deadline, seed=0):
    """
    Optimize the configuration using multi-stage approach.
    """
    # Stage 0: Force Directed Relaxation
    pos_fd, rad_fd = _force_directed_relax(pos, radii, deadline)
    pos_pp_fd, rad_pp_fd = _post_process(pos_fd, rad_fd)
    valid_fd, _ = _vectorized_check(pos_pp_fd, rad_pp_fd)
    if valid_fd:
        score_fd = np.sum(rad_pp_fd)
        if score_fd > np.sum(radii):
            pos = pos_pp_fd
            radii = rad_pp_fd

    x0 = np.concatenate([pos.flatten(), radii])

    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.0, 0.5))

    best_x = x0.copy()
    best_score = np.sum(radii)

    # Stage 1: L-BFGS-B with moderate penalty
    if time.time() < deadline - 20:
        p1 = 150.0
        beta1 = 60.0
        try:
            res1 = minimize(_penalty_function, x0, method='L-BFGS-B', bounds=bounds,
                            args=(p1, beta1),
                            options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'disp': False})
            if res1.fun < _penalty_function(x0, p1, beta1):
                x_curr = res1.x
                pos_curr = x_curr[:2 * N].reshape(N, 2)
                rad_curr = np.maximum(x_curr[2 * N:], 1e-7)
                pos_pp, rad_pp = _post_process(pos_curr, rad_curr)
                valid, _ = _vectorized_check(pos_pp, rad_pp)
                if valid:
                    score = np.sum(rad_pp)
                    if score > best_score:
                        best_score = score
                        best_x = np.concatenate([pos_pp.flatten(), rad_pp])
        except Exception:
            pass

    # Stage 2: L-BFGS-B with higher penalty
    if time.time() < deadline - 15:
        p2 = 500.0
        beta2 = 100.0
        try:
            res2 = minimize(_penalty_function, best_x, method='L-BFGS-B', bounds=bounds,
                            args=(p2, beta2),
                            options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'disp': False})
            if res2.fun < _penalty_function(best_x, p2, beta2):
                x_curr = res2.x
                pos_curr = x_curr[:2 * N].reshape(N, 2)
                rad_curr = np.maximum(x_curr[2 * N:], 1e-7)
                pos_pp, rad_pp = _post_process(pos_curr, rad_curr)
                valid, _ = _vectorized_check(pos_pp, rad_pp)
                if valid:
                    score = np.sum(rad_pp)
                    if score > best_score:
                        best_score = score
                        best_x = np.concatenate([pos_pp.flatten(), rad_pp])
        except Exception:
            pass

    # Stage 3: SLSQP with hard constraints
    if time.time() < deadline - 10:
        try:
            def constraint_boundary(x):
                p = x[:2 * N].reshape(N, 2)
                r = x[2 * N:]
                c1 = p[:, 0] - r
                c2 = 1.0 - p[:, 0] - r
                c3 = p[:, 1] - r
                c4 = 1.0 - p[:, 1] - r
                return np.concatenate([c1, c2, c3, c4])

            def constraint_overlap(x):
                p = x[:2 * N].reshape(N, 2)
                r = x[2 * N:]
                dx = p[_II, 0] - p[_JJ, 0]
                dy = p[_II, 1] - p[_JJ, 1]
                dist_sq = dx * dx + dy * dy
                min_dist = r[_II] + r[_JJ]
                return dist_sq - min_dist * min_dist

            def objective(x):
                return -np.sum(x[2 * N:])

            constraints = [
                {'type': 'ineq', 'fun': constraint_boundary},
                {'type': 'ineq', 'fun': constraint_overlap}
            ]

            res3 = minimize(objective, best_x, method='SLSQP', bounds=bounds,
                            constraints=constraints,
                            options={'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-14, 'disp': False})

            if res3.success or (res3.fun < objective(best_x)):
                pos_curr = res3.x[:2 * N].reshape(N, 2)
                rad_curr = np.maximum(res3.x[2 * N:], 1e-7)
                pos_pp, rad_pp = _post_process(pos_curr, rad_curr)
                valid, _ = _vectorized_check(pos_pp, rad_pp)
                if valid:
                    score = np.sum(rad_pp)
                    if score > best_score:
                        best_score = score
                        best_x = np.concatenate([pos_pp.flatten(), rad_pp])
        except Exception:
            pass

    # Stage 4: Fast local search
    if time.time() < deadline - 5:
        pos_curr = best_x[:2 * N].reshape(N, 2)
        rad_curr = best_x[2 * N:]
        try:
            pos_ls, rad_ls = _local_search(pos_curr, rad_curr, deadline, max_iter=8000)
            valid, _ = _vectorized_check(pos_ls, rad_ls)
            if valid:
                score = np.sum(rad_ls)
                if score > best_score:
                    best_score = score
                    best_x = np.concatenate([pos_ls.flatten(), rad_ls])
        except Exception:
            pass

    pos_out = best_x[:2 * N].reshape(N, 2)
    rad_out = best_x[2 * N:]
    return pos_out, rad_out


def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # Safety margin

    best_pos = None
    best_rad = None
    best_score = 0.0

    # Generate diverse initializations
    inits = []

    # 1. Best Fit seed 1
    pos_bf, rad_bf = _best_fit_init(seed=1)
    inits.append((pos_bf, rad_bf))

    # 2. Best Fit seed 42
    pos_bf2, rad_bf2 = _best_fit_init(seed=42)
    inits.append((pos_bf2, rad_bf2))

    # 3. Hexagonal
    pos_hex, rad_hex = _hex_init()
    inits.append((pos_hex, rad_hex))

    # 4. Grid
    pos_grid, rad_grid = _grid_init()
    inits.append((pos_grid, rad_grid))

    # Optimize each initialization
    for i, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 35:
            break

        pos_pp, rad_pp = _post_process(pos_init, rad_init)
        valid, _ = _vectorized_check(pos_pp, rad_pp)
        if valid:
            score = np.sum(rad_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_rad = rad_pp.copy()

        try:
            pos_opt, rad_opt = _optimize(pos_pp, rad_pp, deadline, seed=i)
            pos_pp_opt, rad_pp_opt = _post_process(pos_opt, rad_opt)
            valid, _ = _vectorized_check(pos_pp_opt, rad_pp_opt)
            if valid:
                score = np.sum(rad_pp_opt)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp_opt.copy()
                    best_rad = rad_pp_opt.copy()
        except Exception:
            pass

    # Final high-precision polish on the best solution
    if best_pos is not None and time.time() < deadline - 5:
        try:
            pos_pol, rad_pol = _polish(best_pos, best_rad, deadline)
            pos_pol, rad_pol = _post_process(pos_pol, rad_pol)
            valid, _ = _vectorized_check(pos_pol, rad_pol)
            if valid:
                score = np.sum(rad_pol)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pol.copy()
                    best_rad = rad_pol.copy()
        except Exception:
            pass

    # Final local search polish
    if best_pos is not None and time.time() < deadline - 3:
        try:
            pos_ls, rad_ls = _local_search(best_pos, best_rad, deadline, max_iter=5000)
            valid, _ = _vectorized_check(pos_ls, rad_ls)
            if valid:
                score = np.sum(rad_ls)
                if score > best_score:
                    best_score = score
                    best_pos = pos_ls.copy()
                    best_rad = rad_ls.copy()
        except Exception:
            pass

    # Fallback
    if best_pos is None:
        pos = np.zeros((N, 2))
        rad = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    rad[k] = 0.095
                    k += 1
        best_pos, best_rad = _post_process(pos, rad)
        best_score = np.sum(best_rad)

    best_pos, best_rad = _post_process(best_pos, best_rad)

    result = np.column_stack([best_pos, best_rad])
    return result

# EVOLVE-BLOCK-END
