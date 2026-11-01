"""Model-and-optimize approach for 50 equal circles in a unit square."""
import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import cKDTree

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill

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
    
    # Use cKDTree for efficient nearest-neighbor distance
    tree = cKDTree(centers)
    # Query each point's nearest neighbor (excluding itself)
    dists, idx = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0
    
    return min(min_pair, min_wall)

def _soft_min_objective(x, tau=0.008):
    """
    Objective: minimize -soft_min(minimum distance to constraint boundary).
    x: flattened array of 2*N coordinates
    """
    centers = x.reshape(-1, 2)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    # Pairwise distances using cKDTree
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    pair_dists = dists[:, 1] / 2.0
    
    # Combine all constraint distances
    all_dists = np.concatenate([walls, pair_dists])
    
    # Soft-min using log-sum-exp
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    log_sum = np.log(np.sum(np.exp(-shifted / tau)))
    soft_min = min_d - tau * log_sum
    
    return -soft_min

def _grad_soft_min(x, tau=0.008):
    """
    Compute gradient of -soft_min objective analytically.
    """
    centers = x.reshape(-1, 2)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    # Upper triangle pairwise distances
    iu, ju = np.triu_indices(N, k=1)
    pair_upper = pair_dists[iu, ju]
    
    all_dists = np.concatenate([walls, pair_upper])
    
    # Soft-min weights: w_k = exp(-d_k/tau) / sum(exp(-d_j/tau))
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    exp_vals = np.exp(-shifted / tau)
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros_like(x)
    
    # Wall gradient contributions
    for i in range(N):
        w_wall = weights[i]
        if walls[i] == wall_dists_x[i]:
            grad[2*i] += w_wall * np.sign(centers[i, 0])
        if walls[i] == wall_dists_y[i]:
            grad[2*i+1] += w_wall * np.sign(centers[i, 1])
    
    # Pairwise gradient contributions
    for k in range(len(iu)):
        w_pair = weights[N + k]
        i, j = iu[k], ju[k]
        d = dists[i, j]
        if d > 1e-12:
            factor = 0.5 * w_pair / d
            diff_ij = centers[i] - centers[j]
            grad[2*i:2*i+2] -= factor * diff_ij
            grad[2*j:2*j+2] += factor * diff_ij
    
    return grad

def _initial_grid():
    """Create initial hexagonal grid layout with slight jitter."""
    # Hexagonal packing: offset every other row
    rows = int(np.ceil(np.sqrt(N * 2 / np.sqrt(3))))
    cols = int(np.ceil(N / rows))
    
    # Hexagonal spacing
    dx = 1.0 / (cols + 0.5)
    dy = dx * np.sqrt(3) / 2
    
    points = []
    for row in range(rows):
        y = -0.5 + dy * (row + 0.5)
        if y > 0.5:
            break
        offset = dx / 2 if row % 2 == 1 else 0
        for col in range(cols):
            x = -0.5 + dx * (col + 0.5) + offset
            if x < -0.5 or x > 0.5:
                continue
            points.append([x, y])
            if len(points) >= N:
                break
        if len(points) >= N:
            break
    
    # If we don't have enough, fill in with square grid
    if len(points) < N:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        for y in ticks:
            for x in ticks:
                if len(points) >= N:
                    break
                if not any(np.allclose([x, y], p) for p in points):
                    points.append([x, y])
            if len(points) >= N:
                break
    
    grid = np.array(points[:N])
    
    rng = np.random.default_rng(12345)
    grid = grid + rng.normal(0, 0.003, grid.shape)
    grid = np.clip(grid, -0.49, 0.49)
    
    return grid

def _simulated_annealing(centers, rng, time_budget):
    """
    Simulated annealing using soft-min objective to escape local minima.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    # Use soft-min objective for SA
    tau_start = 0.02
    tau_end = 0.003
    n_steps = 30000
    step_size_start = 0.05
    step_size_end = 0.002
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        # Annealing schedule
        alpha = step / n_steps
        tau = tau_start * (tau_end / tau_start) ** alpha
        step_size = step_size_start * (step_size_end / step_size_start) ** alpha
        
        # Evaluate current objective
        obj_curr = _soft_min_objective(current.flatten(), tau)
        
        # Generate perturbation: random Gaussian move
        perturbation = rng.normal(0, step_size, current.shape)
        proposed = current + perturbation
        proposed = np.clip(proposed, -0.5, 0.5)
        
        # Evaluate proposed objective
        obj_prop = _soft_min_objective(proposed.flatten(), tau)
        
        # Accept or reject
        if obj_prop < obj_curr:
            current = proposed
        else:
            delta_obj = obj_prop - obj_curr
            if delta_obj < 0:
                accept_prob = np.exp(-delta_obj / (tau * 10))
            else:
                accept_prob = np.exp(-delta_obj / (tau * 10))
            if rng.random() < accept_prob:
                current = proposed
        
        # Track best
        if step % 50 == 0:
            r_curr = _compute_radius(current)
            if r_curr > best_r:
                best_r = r_curr
                best = current.copy()
    
    # Final check
    r_curr = _compute_radius(current)
    if r_curr > best_r:
        best_r = r_curr
        best = current.copy()
    
    return best, best_r

def _lbfgs_optimize(x0, tau=0.008, time_budget=20.0):
    """Run L-BFGS-B optimization on the soft-min objective."""
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
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12,
                     'maxfun': 2000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)

def _high_precision_polish(centers, time_budget):
    """Final high-precision polish using very tight tolerances."""
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    tau = 0.001
    
    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        f = _soft_min_objective(x, tau)
        g = _grad_soft_min(x, tau)
        return f, g
    
    try:
        result = opt.minimize(
            obj_and_grad, current.flatten(), jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-15,
                     'maxfun': 5000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        
        r_polished = _compute_radius(polished)
        if r_polished > best_r:
            best_r = r_polished
            best = polished
    except Exception:
        pass
    
    return best, best_r

def _binary_search_radius(centers, lo, hi, time_budget):
    """Binary search for maximum feasible radius with gradient polishing."""
    start_time = time.time()
    best = centers.copy()
    best_r = lo
    
    for _ in range(15):
        if time.time() - start_time > time_budget:
            break
        
        mid = (lo + hi) / 2
        trial = best.copy()
        
        x = trial.flatten().copy()
        lr = 0.001
        for _ in range(200):
            if time.time() - start_time > time_budget:
                break
            c = x.reshape(-1, 2)
            
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            pair_viol = np.sum(np.maximum(0, mid - dists / 2.0))
            
            wall_x = 0.5 - np.abs(c[:, 0])
            wall_y = 0.5 - np.abs(c[:, 1])
            wall_viol = np.sum(np.maximum(0, mid - np.minimum(wall_x, wall_y)))
            
            v = pair_viol + wall_viol
            if v < 1e-14:
                break
            
            grad = np.zeros_like(x)
            for i in range(N):
                for j in range(i+1, N):
                    d = dists[i, j]
                    if d < 2 * mid and d > 1e-12:
                        factor = 0.5 / d
                        diff_ij = c[i] - c[j]
                        grad[2*i:2*i+2] -= factor * diff_ij
                        grad[2*j:2*j+2] += factor * diff_ij
                if wall_x[i] < mid:
                    grad[2*i] += np.sign(c[i, 0])
                if wall_y[i] < mid:
                    grad[2*i+1] += np.sign(c[i, 1])
            
            x -= lr * grad
            x = np.clip(x, -0.5, 0.5)
        
        trial = x.reshape(-1, 2)
        trial_r = _compute_radius(trial)
        
        if trial_r >= mid - 1e-10:
            lo = mid
            best = trial
            best_r = trial_r
        else:
            hi = mid
    
    return best, best_r

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    best_centers = _initial_grid()
    best_radius = _compute_radius(best_centers)
    
    rng = np.random.default_rng(777)
    
    # Phase 1: Simulated Annealing
    if time.time() < deadline - 60:
        sa_budget = min(45.0, deadline - time.time() - 60)
        if sa_budget > 5:
            best_centers, best_radius = _simulated_annealing(best_centers, rng, sa_budget)
    
    # Phase 2: L-BFGS with multiple tau values
    if time.time() < deadline - 35:
        x0 = best_centers.flatten()
        for tau in [0.02, 0.015, 0.01, 0.008, 0.005, 0.003, 0.002]:
            if time.time() > deadline - 20:
                break
            trial = _lbfgs_optimize(x0, tau=tau, time_budget=min(10, deadline - time.time() - 15))
            trial_r = _compute_radius(trial)
            if trial_r > best_radius:
                best_radius = trial_r
                best_centers = trial
            x0 = best_centers.flatten()
    
    # Phase 3: Binary search
    if time.time() < deadline - 15:
        lo = best_radius
        hi = min(0.08, best_radius * 1.03)
        best_centers, best_radius = _binary_search_radius(
            best_centers, lo, hi, time_budget=min(15, deadline - time.time() - 8)
        )
    
    # Phase 4: High-precision polish
    if time.time() < deadline - 5:
        best_centers, best_radius = _high_precision_polish(
            best_centers, time_budget=min(12, deadline - time.time() - 3)
        )
        
        # Final micro-polish with even tighter tolerance
        if time.time() < deadline - 2:
            remaining = deadline - time.time() - 2
            if remaining > 2:
                try:
                    x = best_centers.flatten()
                    result = opt.minimize(
                        lambda v: _soft_min_objective(v, tau=0.001),
                        x, method='L-BFGS-B',
                        bounds=[(-0.5, 0.5)] * (2 * N),
                        options={'maxiter': 200, 'ftol': 1e-20, 'gtol': 1e-16,
                                 'maxfun': 500, 'disp': False}
                    )
                    polished = np.clip(result.x.reshape(-1, 2), -0.5, 0.5)
                    r_polished = _compute_radius(polished)
                    if r_polished > best_radius:
                        best_radius = r_polished
                        best_centers = polished
                except Exception:
                    pass
    
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_grid()
    
    return best_centers
