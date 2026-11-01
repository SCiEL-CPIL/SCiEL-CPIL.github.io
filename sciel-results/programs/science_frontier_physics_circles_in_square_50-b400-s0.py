"""
Optimized packing of 50 equal circles in a unit square.
Combines Simulated Annealing for global search with high-precision L-BFGS polishing
using a full-pairwise smooth violation objective.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
DEADLINE = 147.0  # Safety margin before 150s kill

# Precompute upper-triangle indices for pairwise operations
_IU = np.triu_indices(N, k=1)
_NPAIRS = len(_IU[0])

def _compute_radius(centers):
    """Compute the true maximum radius for a given configuration."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    if min_wall < 0:
        return 0.0
    
    # Pairwise distances (Full N x N for accuracy)
    diff = centers[:, None, :] - centers[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair = np.sqrt(np.min(dist_sq)) / 2.0
    
    return min(min_pair, min_wall)

def _soft_violation_obj(x, r_target, beta):
    """
    Objective: Minimize smooth violation for target radius r_target.
    Violation for pair (i,j): softplus(beta * (2*r_target - d_ij))
    Violation for wall i: softplus(beta * (r_target - w_i))
    Vectorized over pairs.
    """
    centers = x.reshape(-1, 2)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    wall_viol = r_target - walls
    
    # Pairwise distances (vectorized via upper-triangle indices)
    ci = centers[_IU[0]]
    cj = centers[_IU[1]]
    diff = ci - cj
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    pair_viol = 2.0 * r_target - dists
    
    # Softplus: log(1 + exp(z)) / beta
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
    
    # Wall gradients
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    wall_viol = r_target - walls
    
    z_wall = np.clip(beta * wall_viol, -50, 50)
    sig_wall = 1.0 / (1.0 + np.exp(-z_wall))
    
    # Vectorized wall gradient contribution
    wx = wall_dists_x < wall_dists_y
    wy = wall_dists_y < wall_dists_x
    wt = ~(wx | wy)
    
    grad[2*np.nonzero(wx)[0]] += sig_wall[wx] * np.sign(centers[wx, 0])
    grad[2*np.nonzero(wy)[0]+1] += sig_wall[wy] * np.sign(centers[wy, 1])
    if np.any(wt):
        idx_wt = np.nonzero(wt)[0]
        grad[2*idx_wt] += 0.5 * sig_wall[wt] * np.sign(centers[idx_wt, 0])
        grad[2*idx_wt+1] += 0.5 * sig_wall[wt] * np.sign(centers[idx_wt, 1])
    
    # Pair gradients (vectorized)
    ci = centers[_IU[0]]
    cj = centers[_IU[1]]
    diff_vec = ci - cj  # shape (_NPAIRS, 2)
    dists = np.sqrt(np.sum(diff_vec ** 2, axis=1))
    dists_safe = np.maximum(dists, 1e-12)
    pair_viol = 2.0 * r_target - dists
    
    z_pair = np.clip(beta * pair_viol, -50, 50)
    sig_pair = 1.0 / (1.0 + np.exp(-z_pair))
    
    # factor_k = sig_pair[k] / dists[k], shape (_NPAIRS,)
    factor = sig_pair / dists_safe
    # contribution per pair: factor * diff_vec  shape (_NPAIRS, 2)
    contrib = factor[:, None] * diff_vec
    
    # Scatter: for pair (i,j), grad_i -= contrib, grad_j += contrib
    np.add.at(grad, 2 * _IU[0], -contrib[:, 0])
    np.add.at(grad, 2 * _IU[0] + 1, -contrib[:, 1])
    np.add.at(grad, 2 * _IU[1], contrib[:, 0])
    np.add.at(grad, 2 * _IU[1] + 1, contrib[:, 1])
    
    return grad

def _initial_grid():
    """Create initial grid layout with slight jitter."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N].copy()
    
    rng = np.random.default_rng(12345)
    grid = grid + rng.normal(0, 0.005, grid.shape)
    grid = np.clip(grid, -0.49, 0.49)
    
    return grid

def _hex_init():
    """Create a hexagonal packing initial layout."""
    best_grid = None
    best_r = 0
    
    rng = np.random.default_rng(42)
    
    for attempt in range(5):
        rows = rng.integers(5, 9)
        cols = int(np.ceil(N / rows))
        
        s_width = 1.0 / (cols + 0.5)
        s_height = 1.0 / (rows * 0.866 + 0.5)
        s = min(s_width, s_height)
        
        centers = []
        for j in range(rows):
            for i in range(cols):
                x = (i - (cols - 1) / 2) * s
                y = (j - (rows - 1) / 2) * s * 0.866
                if j % 2 == 1:
                    x += s / 2
                centers.append([x, y])
        
        centers = np.array(centers[:N])
        if len(centers) < N:
            while len(centers) < N:
                centers = np.vstack([centers, rng.uniform(-0.45, 0.45, 2)])
        
        centers = np.clip(centers, -0.49, 0.49)
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_grid = centers.copy()
    
    return best_grid if best_grid is not None else _initial_grid()

def _random_init():
    """Create a random initial layout."""
    rng = np.random.default_rng(999)
    centers = rng.uniform(-0.45, 0.45, (N, 2))
    return centers

def _simulated_annealing(centers, rng, time_budget, r_target=None):
    """
    Simulated annealing to optimize the minimum radius directly.
    Uses vectorized radius computation (no per-step KDTree build).
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    # If r_target is provided, we minimize violation instead
    use_violation = r_target is not None
    
    T_start = 0.02
    T_end = 0.0001
    n_steps = 40000
    cooling = (T_end / T_start) ** (1.0 / max(n_steps, 1))
    T = T_start
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        # Quick vectorized radius check
        wall_x = 0.5 - np.abs(current[:, 0])
        wall_y = 0.5 - np.abs(current[:, 1])
        min_wall = np.min(np.minimum(wall_x, wall_y))
        
        # Vectorized pairwise min distance
        d = current[:, None, :] - current[None, :, :]
        dist_sq = np.einsum('ijk,ijk->ij', d, d)
        np.fill_diagonal(dist_sq, np.inf)
        min_pair = np.sqrt(np.min(dist_sq)) / 2.0
        curr_r = min(min_pair, min_wall)
        
        # Move generation
        i = rng.integers(0, N)
        move_size = 0.005 + 0.01 * T / T_start
        direction = rng.normal(size=2)
        direction /= np.linalg.norm(direction)
        
        trial = current.copy()
        trial[i] += direction * move_size
        trial = np.clip(trial, -0.5, 0.5)
        
        trial_r = _compute_radius(trial)
        delta_r = trial_r - curr_r
        
        if use_violation:
            # In violation mode, we just track the best radius found
            # but accept based on radius improvement mostly
            pass
            
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

def _lbfgs_feasibility_polish(x0, r_target, beta, time_budget):
    """
    Run L-BFGS-B to minimize violation for a specific target radius r_target.
    Uses vectorized objective and gradient.
    """
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
            options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-13,
                     'maxfun': 10000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)

def _refine_radius(centers, time_budget):
    """
    Binary search for the maximum feasible radius using L-BFGS polishing.
    Increased iterations and tighter tolerance.
    """
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)
    
    # Initial bounds
    lo = best_r
    hi = min(0.08, best_r * 1.05)
    
    search_steps = 14
    for i in range(search_steps):
        if time.time() - start_time > time_budget:
            break
            
        mid = (lo + hi) / 2
        # Increase beta as we converge to get sharper constraints
        beta = 200.0 + 100.0 * i
        x0 = best.flatten()
        
        remaining = time_budget - (time.time() - start_time)
        if remaining < 5:
            break
        lbfgs_budget = min(12.0, remaining * 0.85)
        
        trial = _lbfgs_feasibility_polish(x0, mid, beta, lbfgs_budget)
        trial_r = _compute_radius(trial)
        
        if trial_r >= mid - 1e-7:
            lo = mid
            best = trial
            best_r = trial_r
        else:
            hi = mid
            
    return best, best_r

def _high_precision_polish(centers, time_budget):
    """
    Final high-precision polish using increasing beta.
    Includes a "push" phase targeting slightly above current best.
    """
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)
    
    # Progressive sharpening of the violation objective
    betas = [500, 1000, 2000, 4000, 8000]
    
    for beta in betas:
        if time.time() - start_time > time_budget:
            break
            
        r_target = best_r
        x0 = best.flatten()
        
        remaining = time_budget - (time.time() - start_time)
        if remaining < 5:
            break
        lbfgs_budget = min(12.0, remaining * 0.85)
        
        trial = _lbfgs_feasibility_polish(x0, r_target, beta, lbfgs_budget)
        trial_r = _compute_radius(trial)
        
        if trial_r > best_r:
            best_r = trial_r
            best = trial
            
    # Push phase: try to target slightly above best_r
    if time.time() - start_time < time_budget - 10:
        for push_frac in [1.001, 1.002, 1.004, 1.007, 1.01]:
            if time.time() - start_time > time_budget - 5:
                break
            r_push = best_r * push_frac
            if r_push > 0.08:
                break
            remaining = time_budget - (time.time() - start_time)
            if remaining < 5:
                break
            lbfgs_budget = min(12.0, remaining * 0.85)
            # Use higher beta for sharper constraint enforcement during push
            beta_push = min(8000.0, 500.0 * (1.0 / (1.0 - push_frac + 1.0)))
            trial = _lbfgs_feasibility_polish(best.flatten(), r_push, beta_push, lbfgs_budget)
            trial_r = _compute_radius(trial)
            if trial_r > best_r:
                best_r = trial_r
                best = trial
    
    return best, best_r

def _local_search(centers, time_budget):
    """
    A simple local search that tries to push circles apart if they are too close.
    This can help escape local minima where SA might get stuck.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    n_iters = 5000
    step_size = 0.01
    
    for it in range(n_iters):
        if time.time() - start_time > time_budget:
            break
            
        curr_r = _compute_radius(current)
        
        # Find the pair with minimum distance
        diff = current[:, None, :] - current[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_idx = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
        i, j = min_idx
        
        # Also check walls
        wall_x = 0.5 - np.abs(current[:, 0])
        wall_y = 0.5 - np.abs(current[:, 1])
        walls = np.minimum(wall_x, wall_y)
        min_wall_idx = np.argmin(walls)
        
        min_pair_dist = np.sqrt(dist_sq[i, j])
        min_wall_dist = walls[min_wall_idx]
        
        # Try to move the most constrained circle
        if min_pair_dist / 2.0 < min_wall_dist:
            # Move i and j apart
            direction = current[i] - current[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction /= norm
                trial = current.copy()
                trial[i] += direction * step_size
                trial[j] -= direction * step_size
                trial = np.clip(trial, -0.5, 0.5)
                trial_r = _compute_radius(trial)
                if trial_r > best_r:
                    best_r = trial_r
                    best = trial.copy()
                    current = trial
        else:
            # Move the circle closest to the wall away from it
            idx = min_wall_idx
            # Determine which wall is closest
            x = current[idx, 0]
            y = current[idx, 1]
            trial = current.copy()
            if 0.5 - x < x + 0.5: # Closer to right
                trial[idx, 0] -= step_size
            else:
                trial[idx, 0] += step_size
            if 0.5 - y < y + 0.5: # Closer to top
                trial[idx, 1] -= step_size
            else:
                trial[idx, 1] += step_size
            trial = np.clip(trial, -0.5, 0.5)
            trial_r = _compute_radius(trial)
            if trial_r > best_r:
                best_r = trial_r
                best = trial.copy()
                current = trial
                
    return best, best_r

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    rng = np.random.default_rng(777)
    
    # 1. Initialization
    inits = [
        _initial_grid(),
        _hex_init(),
        _random_init(),
    ]
    
    best_centers = None
    best_radius = 0
    
    for init in inits:
        r = _compute_radius(init)
        if r > best_radius:
            best_radius = r
            best_centers = init.copy()
            
    if best_centers is None:
        best_centers = _initial_grid()
        best_radius = _compute_radius(best_centers)
        
    # 2. Simulated Annealing (Global Search)
    # Vectorized radius makes SA ~3x faster, so we can use more of the budget
    sa_budget = min(35.0, (deadline - time.time()) * 0.25)
    if sa_budget > 5:
        best_centers, best_radius = _simulated_annealing(best_centers, rng, sa_budget)
        
    # 3. Local Search (to escape shallow local minima)
    if time.time() < deadline - 60:
        best_centers, best_radius = _local_search(
            best_centers, time_budget=min(10.0, deadline - time.time() - 50)
        )
        
    # 4. Feasibility-Driven Radius Refinement (Coarse to Fine)
    if time.time() < deadline - 40:
        best_centers, best_radius = _refine_radius(
            best_centers, time_budget=min(40.0, deadline - time.time() - 20)
        )
        
    # 5. High Precision Polish (Final Tightening + Push)
    if time.time() < deadline - 12:
        best_centers, best_radius = _high_precision_polish(
            best_centers, time_budget=min(30.0, deadline - time.time() - 5)
        )
        
    # Final Validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_grid()
        
    return best_centers
