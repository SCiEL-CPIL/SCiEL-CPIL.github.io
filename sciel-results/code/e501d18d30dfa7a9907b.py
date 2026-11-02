"""
Combined approach: current program's SA + cKDTree machinery
with inspiration's full-pairwise smooth violation objective and SLSQP.
"""
import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import cKDTree

N = 50
DEADLINE = 145.0

# Precompute upper-triangle indices for pairwise operations
_IU = np.triu_indices(N, k=1)
_NPAIRS = len(_IU[0])

def _compute_radius(centers):
    """Compute the true maximum radius for a given configuration."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    if min_wall < 0:
        return 0.0
    
    tree = cKDTree(centers)
    dists, idx = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0
    
    return min(min_pair, min_wall)

def _soft_violation_obj(x, r_target, beta):
    """
    Objective: Minimize smooth violation for target radius r_target.
    Uses full pairwise distances via precomputed indices.
    """
    centers = x.reshape(-1, 2)
    
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    wall_viol = r_target - walls
    
    ci = centers[_IU[0]]
    cj = centers[_IU[1]]
    diff = ci - cj
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    pair_viol = 2.0 * r_target - dists
    
    z_wall = np.clip(beta * wall_viol, -50, 50)
    z_pair = np.clip(beta * pair_viol, -50, 50)
    
    soft_wall = np.log1p(np.exp(z_wall)) / beta
    soft_pair = np.log1p(np.exp(z_pair)) / beta
    
    return float(np.sum(soft_wall) + np.sum(soft_pair))

def _soft_violation_grad(x, r_target, beta):
    """
    Gradient of the soft violation objective. Fully vectorized.
    """
    centers = x.reshape(-1, 2)
    grad = np.zeros_like(x)
    
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    wall_viol = r_target - walls
    
    z_wall = np.clip(beta * wall_viol, -50, 50)
    sig_wall = 1.0 / (1.0 + np.exp(-z_wall))
    
    wx = wall_dists_x < wall_dists_y
    wy = wall_dists_y < wall_dists_x
    wt = ~(wx | wy)
    
    idx_wx = np.nonzero(wx)[0]
    idx_wy = np.nonzero(wy)[0]
    idx_wt = np.nonzero(wt)[0]
    
    if len(idx_wx) > 0:
        grad[2*idx_wx] += sig_wall[wx] * np.sign(centers[idx_wx, 0])
    if len(idx_wy) > 0:
        grad[2*idx_wy+1] += sig_wall[wy] * np.sign(centers[idx_wy, 1])
    if len(idx_wt) > 0:
        grad[2*idx_wt] += 0.5 * sig_wall[wt] * np.sign(centers[idx_wt, 0])
        grad[2*idx_wt+1] += 0.5 * sig_wall[wt] * np.sign(centers[idx_wt, 1])
    
    ci = centers[_IU[0]]
    cj = centers[_IU[1]]
    diff_vec = ci - cj
    dists = np.sqrt(np.sum(diff_vec ** 2, axis=1))
    dists_safe = np.maximum(dists, 1e-12)
    pair_viol = 2.0 * r_target - dists
    
    z_pair = np.clip(beta * pair_viol, -50, 50)
    sig_pair = 1.0 / (1.0 + np.exp(-z_pair))
    
    factor = sig_pair / dists_safe
    contrib = factor[:, None] * diff_vec
    
    np.add.at(grad, 2 * _IU[0], -contrib[:, 0])
    np.add.at(grad, 2 * _IU[0] + 1, -contrib[:, 1])
    np.add.at(grad, 2 * _IU[1], contrib[:, 0])
    np.add.at(grad, 2 * _IU[1] + 1, contrib[:, 1])
    
    return grad

def _soft_min_objective(x, tau=0.008, k_neighbors=5):
    """
    Objective: minimize -soft_min(minimum distance to constraint boundary).
    Uses top-k nearest neighbors for a better approximation of the true min distance.
    """
    centers = x.reshape(-1, 2)
    
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    tree = cKDTree(centers)
    k = min(k_neighbors + 1, N)
    dists, idxs = tree.query(centers, k=k)
    pair_dists = dists[:, 1] / 2.0
    
    if k >= 3:
        pair_dists_2nd = dists[:, 2] / 2.0
        all_pair_dists = np.concatenate([pair_dists, pair_dists_2nd])
    else:
        all_pair_dists = pair_dists
    
    all_dists = np.concatenate([walls, all_pair_dists])
    
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    log_sum = np.log(np.sum(np.exp(-shifted / tau)))
    soft_min = min_d - tau * log_sum
    
    return -soft_min

def _grad_soft_min(x, tau=0.008, k_neighbors=5):
    """
    Compute gradient of -soft_min objective analytically using cKDTree for speed.
    """
    centers = x.reshape(-1, 2)
    
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    tree = cKDTree(centers)
    k = min(k_neighbors + 1, N)
    dists, idxs = tree.query(centers, k=k)
    
    pair_dists_list = []
    pair_indices_list = []
    for i in range(N):
        for j_idx in range(1, k):
            j = idxs[i, j_idx]
            d = dists[i, j_idx]
            if d > 1e-12:
                pair_dists_list.append(d / 2.0)
                pair_indices_list.append((i, j))
    
    all_pair_dists = np.array(pair_dists_list) if pair_dists_list else np.array([0.0])
    
    all_dists = np.concatenate([walls, all_pair_dists])
    
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    exp_vals = np.exp(-shifted / tau)
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros_like(x)
    
    for i in range(N):
        w_wall = weights[i]
        if w_wall < 1e-15:
            continue
        if wall_dists_x[i] <= wall_dists_y[i]:
            grad[2*i] += w_wall * np.sign(centers[i, 0])
        else:
            grad[2*i+1] += w_wall * np.sign(centers[i, 1])
    
    for idx, (i, j) in enumerate(pair_indices_list):
        w_pair = weights[N + idx]
        if w_pair < 1e-15:
            continue
        d = all_pair_dists[idx] * 2.0
        if d > 1e-12:
            factor = 0.5 * w_pair / d
            diff_ij = centers[i] - centers[j]
            grad[2*i:2*i+2] -= factor * diff_ij
            grad[2*j:2*j+2] += factor * diff_ij
    
    return grad

def _initial_grid():
    """Create initial hexagonal packing layout with slight jitter."""
    row_counts = [8, 7, 8, 7, 8, 7, 5]
    
    points = []
    total_rows = len(row_counts)
    vert_spacing = 0.9 / (total_rows - 1) if total_rows > 1 else 0
    
    for row_idx, count in enumerate(row_counts):
        y = -0.45 + row_idx * vert_spacing
        horiz_spacing = 0.9 / count if count > 1 else 0
        x_start = -0.45 + (0.9 - horiz_spacing * (count - 1)) / 2
        if row_idx % 2 == 1:
            x_offset = horiz_spacing / 2
        else:
            x_offset = 0
        
        for i in range(count):
            x = x_start + i * horiz_spacing + x_offset
            points.append([x, y])
    
    points = np.array(points[:N])
    
    rng = np.random.default_rng(12345)
    points = points + rng.normal(0, 0.008, points.shape)
    points = np.clip(points, -0.49, 0.49)
    
    return points

def _simulated_annealing(centers, rng, time_budget):
    """
    Simulated annealing to optimize the minimum radius directly.
    Uses cKDTree for fast distance queries and biased moves toward constrained circles.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    T_start = 0.03
    T_end = 1e-7
    n_steps = 80000
    cooling = (T_end / T_start) ** (1.0 / max(n_steps, 1))
    T = T_start
    
    trial = current.copy()
    
    tree = cKDTree(current)
    last_tree_build = 0
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        if step % 50 == 0 or step == 0:
            tree = cKDTree(current)
            last_tree_build = step
        
        dists, _ = tree.query(current, k=3)
        min_pair = np.min(dists[:, 1]) / 2.0
        wall_x = 0.5 - np.abs(current[:, 0])
        wall_y = 0.5 - np.abs(current[:, 1])
        min_wall = np.min(np.minimum(wall_x, wall_y))
        curr_r = min(min_pair, min_wall)
        
        all_dists = np.concatenate([np.minimum(wall_x, wall_y), dists[:, 1] / 2.0])
        min_d = np.min(all_dists)
        weights = np.exp(-(all_dists - min_d) / 0.004)
        weights /= np.sum(weights)
        
        idx = rng.choice(len(all_dists), p=weights)
        
        move_size = 0.003 + 0.02 * T / T_start
        
        if idx < N:
            i = idx
            direction = rng.integers(0, 4)
            delta = np.zeros(2)
            if direction == 0:
                delta[0] = move_size * rng.uniform(0.5, 1.0)
            elif direction == 1:
                delta[0] = -move_size * rng.uniform(0.5, 1.0)
            elif direction == 2:
                delta[1] = move_size * rng.uniform(0.5, 1.0)
            else:
                delta[1] = -move_size * rng.uniform(0.5, 1.0)
            
            trial = current.copy()
            trial[i] += delta
            trial = np.clip(trial, -0.5, 0.5)
        else:
            i = idx - N
            _, nn_idx = tree.query(current[i], k=2)
            j = nn_idx[1]
            
            diff = current[i] - current[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-9:
                unit = diff / dist
            else:
                unit = rng.normal(size=2)
                unit /= np.linalg.norm(unit)
            
            trial = current.copy()
            frac = rng.uniform(0.3, 0.7)
            trial[i] += unit * move_size * frac
            trial[j] -= unit * move_size * (1.0 - frac)
            trial = np.clip(trial, -0.5, 0.5)
        
        trial_r = _compute_radius(trial)
        delta_r = trial_r - curr_r
        
        if delta_r >= 0:
            current = trial
            if trial_r > best_r:
                best_r = trial_r
                best = trial.copy()
        else:
            if rng.random() < np.exp(delta_r / max(T, 1e-10)):
                current = trial
        
        T *= cooling
    
    return best, best_r

def _lbfgs_soft_violation(x0, r_target, beta, time_budget):
    """Run L-BFGS-B on the soft violation objective for a target radius."""
    start_time = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        f = _soft_violation_obj(x, r_target, beta)
        g = _soft_violation_grad(x, r_target, beta)
        return f, g
    
    try:
        result = opt.minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12,
                     'maxfun': 2000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)

def _lbfgs_soft_min(x0, tau, time_budget):
    """Run L-BFGS-B on the soft-min objective."""
    start_time = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        f = _soft_min_objective(x, tau)
        g = _grad_soft_min(x, tau)
        return f, g
    
    try:
        result = opt.minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-12,
                     'maxfun': 1500, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)

def _slsqp_optimize(x0, r_target, time_budget):
    """
    SLSQP optimization with explicit constraints.
    Constraints: pairwise distances >= 2*r_target, wall distances >= r_target.
    """
    start_time = time.time()
    
    # Objective: minimize 0 (feasibility problem)
    def objective(x):
        if time.time() - start_time > time_budget:
            return 0.0
        return 0.0
    
    # Constraints: each constraint >= 0
    constraints = []
    
    # Wall constraints
    for i in range(N):
        def wall_x_con(x, i=i):
            if time.time() - start_time > time_budget:
                return 1.0
            c = x.reshape(-1, 2)
            return 0.5 - abs(c[i, 0]) - r_target
        def wall_x_grad(x, i=i):
            if time.time() - start_time > time_budget:
                return np.zeros(2*N)
            g = np.zeros(2*N)
            c = x.reshape(-1, 2)
            g[2*i] = -np.sign(c[i, 0]) if c[i, 0] != 0 else -1.0
            return g
        def wall_y_con(x, i=i):
            if time.time() - start_time > time_budget:
                return 1.0
            c = x.reshape(-1, 2)
            return 0.5 - abs(c[i, 1]) - r_target
        def wall_y_grad(x, i=i):
            if time.time() - start_time > time_budget:
                return np.zeros(2*N)
            g = np.zeros(2*N)
            c = x.reshape(-1, 2)
            g[2*i+1] = -np.sign(c[i, 1]) if c[i, 1] != 0 else -1.0
            return g
        
        constraints.append({'type': 'ineq', 'fun': wall_x_con, 'jac': wall_x_grad})
        constraints.append({'type': 'ineq', 'fun': wall_y_con, 'jac': wall_y_grad})
    
    # Pairwise constraints (only nearby pairs to keep it manageable)
    # Use all pairs but limit to top constraints
    for idx in range(_NPAIRS):
        i = _IU[0][idx]
        j = _IU[1][idx]
        
        def pair_con(x, i=i, j=j):
            if time.time() - start_time > time_budget:
                return 1.0
            c = x.reshape(-1, 2)
            diff = c[i] - c[j]
            return np.linalg.norm(diff) - 2.0 * r_target
        def pair_grad(x, i=i, j=j):
            if time.time() - start_time > time_budget:
                return np.zeros(2*N)
            g = np.zeros(2*N)
            c = x.reshape(-1, 2)
            diff = c[i] - c[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-12:
                unit = diff / dist
                g[2*i:2*i+2] = unit
                g[2*j:2*j+2] = -unit
            return g
        
        constraints.append({'type': 'ineq', 'fun': pair_con, 'jac': pair_grad})
    
    try:
        result = opt.minimize(
            objective, x0, method='SLSQP',
            bounds=[(-0.5, 0.5)] * (2 * N),
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
        )
        if result.success:
            polished = result.x.reshape(-1, 2)
            polished = np.clip(polished, -0.5, 0.5)
            return polished
        else:
            return x0.reshape(-1, 2)
    except Exception:
        return x0.reshape(-1, 2)

def _contact_polish(centers, time_budget):
    """
    Contact-enforcement polish: iteratively identify touching pairs/walls
    and push them apart to maximize the minimum distance.
    """
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)
    
    step = 0.002
    no_improve_count = 0
    
    for iteration in range(300):
        if time.time() - start_time > time_budget:
            break
        
        c = best.copy()
        tree = cKDTree(c)
        dists, idxs = tree.query(c, k=4)
        min_pair_d = np.min(dists[:, 1])
        wall_x = 0.5 - np.abs(c[:, 0])
        wall_y = 0.5 - np.abs(c[:, 1])
        min_wall_d = np.min(np.minimum(wall_x, wall_y))
        curr_r = min(min_pair_d / 2.0, min_wall_d)
        
        threshold = curr_r * 1.001
        
        pair_moves = []
        for i in range(N):
            for j_idx in range(1, 4):
                j = idxs[i, j_idx]
                if j < i:
                    continue
                d = dists[i, j_idx]
                if d / 2.0 < threshold:
                    diff = c[i] - c[j]
                    dist = np.linalg.norm(diff)
                    if dist > 1e-12:
                        unit = diff / dist
                        pair_moves.append((i, j, unit, (d/2.0 - curr_r) + 0.001))
        
        wall_moves = []
        walls = np.minimum(wall_x, wall_y)
        for i in range(N):
            if walls[i] < threshold:
                if wall_x[i] <= wall_y[i]:
                    wall_moves.append((i, 0, np.sign(c[i, 0]) if c[i, 0] != 0 else 1.0, (walls[i] - curr_r) + 0.001))
                else:
                    wall_moves.append((i, 1, np.sign(c[i, 1]) if c[i, 1] != 0 else 1.0, (walls[i] - curr_r) + 0.001))
        
        if not pair_moves and not wall_moves:
            break
        
        total_weight = sum(m[3] for m in pair_moves) + sum(m[3] for m in wall_moves)
        if total_weight < 1e-10:
            break
        
        for i, j, unit, weight in pair_moves:
            frac = 0.5 * step * weight / total_weight
            c[i] += unit * frac
            c[j] -= unit * frac
        
        for i, axis, sign, weight in wall_moves:
            move = step * weight / total_weight
            if axis == 0:
                c[i, 0] += sign * move
            else:
                c[i, 1] += sign * move
        
        c = np.clip(c, -0.5, 0.5)
        r = _compute_radius(c)
        
        if r > best_r:
            best_r = r
            best = c
            no_improve_count = 0
            step = min(step * 1.1, 0.005)
        else:
            no_improve_count += 1
            step = max(step * 0.8, 0.0005)
            if no_improve_count > 30:
                break
    
    return best, best_r

def _high_precision_polish(centers, time_budget):
    """
    High-precision polish using multiple optimization methods with tight tolerances.
    """
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)
    
    # Phase A: L-BFGS with soft-min at progressively smaller tau
    for tau in [0.004, 0.003, 0.002, 0.001]:
        if time.time() - start_time > time_budget * 0.4:
            break
        
        x0 = best.flatten().copy()
        trial = _lbfgs_soft_min(x0, tau, time_budget=min(6, time_budget - (time.time() - start_time) - 2))
        trial_r = _compute_radius(trial)
        if trial_r > best_r:
            best_r = trial_r
            best = trial
    
    # Phase B: SLSQP with target radius slightly above current best
    if time.time() - start_time < time_budget * 0.7:
        r_target = best_r * 1.001
        x0 = best.flatten().copy()
        trial = _slsqp_optimize(x0, r_target, time_budget=min(10, time_budget - (time.time() - start_time) - 3))
        trial_r = _compute_radius(trial)
        if trial_r > best_r:
            best_r = trial_r
            best = trial
    
    # Phase C: L-BFGS with soft violation at target radius
    if time.time() - start_time < time_budget * 0.85:
        r_target = best_r * 1.002
        x0 = best.flatten().copy()
        for beta in [50, 100, 200]:
            if time.time() - start_time > time_budget * 0.9:
                break
            trial = _lbfgs_soft_violation(x0, r_target, beta, time_budget=min(5, time_budget - (time.time() - start_time) - 2))
            trial_r = _compute_radius(trial)
            if trial_r > best_r:
                best_r = trial_r
                best = trial
            x0 = best.flatten().copy()
    
    # Final contact polish with remaining time
    remaining = time_budget - (time.time() - start_time)
    if remaining > 2:
        best, best_r = _contact_polish(best, time_budget=min(remaining * 0.5, 8.0))
    
    return best, best_r

def _binary_search_radius(centers, lo, hi, time_budget):
    """Binary search for maximum feasible radius with gradient polishing."""
    start_time = time.time()
    best = centers.copy()
    best_r = lo
    
    for _ in range(12):
        if time.time() - start_time > time_budget:
            break
        
        mid = (lo + hi) / 2
        trial = best.copy()
        
        # Use soft violation L-BFGS to try to achieve radius mid
        x0 = trial.flatten().copy()
        trial = _lbfgs_soft_violation(x0, mid, beta=100, time_budget=min(4, time_budget - (time.time() - start_time) - 1))
        
        trial_r = _compute_radius(trial)
        
        if trial_r >= mid - 1e-9:
            lo = mid
            best = trial
            best_r = trial_r
        else:
            hi = mid
    
    return best, best_r

def _restart_and_optimize(rng, time_budget):
    """
    Generate a new random initial configuration, run a short SA + LBFGS cycle,
    and return the best configuration found.
    """
    start_time = time.time()
    
    choice = rng.random()
    if choice < 0.4:
        centers = _initial_grid()
        centers += rng.normal(0, 0.02, centers.shape)
    elif choice < 0.7:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        centers = grid[:N].copy()
        centers += rng.normal(0, 0.015, centers.shape)
    else:
        centers = rng.uniform(-0.45, 0.45, (N, 2))
    
    centers = np.clip(centers, -0.49, 0.49)
    curr_r = _compute_radius(centers)
    
    sa_time = min(12.0, (time_budget - (time.time() - start_time)) * 0.5)
    if sa_time > 3:
        centers, curr_r = _simulated_annealing(centers, rng, sa_time)
    
    lbfgs_time = min(15.0, time_budget - (time.time() - start_time) - 5)
    if lbfgs_time > 3:
        x0 = centers.flatten()
        for tau in [0.01, 0.005]:
            if time.time() - start_time > time_budget - 3:
                break
            trial = _lbfgs_soft_min(x0, tau, time_budget=min(6, time_budget - (time.time() - start_time) - 3))
            trial_r = _compute_radius(trial)
            if trial_r > curr_r:
                curr_r = trial_r
                centers = trial
            x0 = centers.flatten()
    
    return centers, curr_r

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_centers = _initial_grid()
    best_radius = _compute_radius(best_centers)
    
    rng = np.random.default_rng(777)
    
    # Phase 1: Simulated Annealing
    sa_budget = min(35.0, (deadline - time.time()) * 0.25)
    if sa_budget > 5:
        best_centers, best_radius = _simulated_annealing(best_centers, rng, sa_budget)
    
    # Phase 2: L-BFGS soft-min optimization
    if time.time() < deadline - 40:
        x0 = best_centers.flatten()
        for tau in [0.01, 0.008, 0.005]:
            if time.time() > deadline - 30:
                break
            trial = _lbfgs_soft_min(x0, tau, time_budget=min(7, deadline - time.time() - 15))
            trial_r = _compute_radius(trial)
            if trial_r > best_radius:
                best_radius = trial_r
                best_centers = trial
            x0 = best_centers.flatten()
    
    # Phase 3: High-precision polish (SLSQP + soft violation + contact)
    if time.time() < deadline - 25:
        best_centers, best_radius = _high_precision_polish(
            best_centers, time_budget=min(20, deadline - time.time() - 10)
        )
    
    # Phase 4: Restart with different initializations
    restarts = 3
    for r in range(restarts):
        if time.time() > deadline - 25:
            break
        
        restart_time = min(15.0, (deadline - time.time()) / (restarts - r) - 5)
        if restart_time < 5:
            break
        
        new_centers, new_radius = _restart_and_optimize(rng, restart_time)
        if new_radius > best_radius:
            best_radius = new_radius
            best_centers = new_centers.copy()
    
    # Phase 5: Final high-precision polish
    if time.time() < deadline - 12:
        best_centers, best_radius = _high_precision_polish(
            best_centers, time_budget=min(10, deadline - time.time() - 6)
        )
    
    # Phase 6: Binary search with soft violation
    if time.time() < deadline - 8:
        lo = best_radius
        hi = min(0.08, best_radius * 1.02)
        best_centers, best_radius = _binary_search_radius(
            best_centers, lo, hi, time_budget=min(6, deadline - time.time() - 3)
        )
    
    # Final validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_grid()
    
    return best_centers
