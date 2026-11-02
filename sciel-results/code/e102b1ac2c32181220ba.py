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

def _active_constraints(centers, r, tol=1e-6):
    """Identify which constraints are active (tight) at radius r."""
    walls_x = 0.5 - np.abs(centers[:, 0])
    walls_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(walls_x, walls_y)
    
    tree = cKDTree(centers)
    dists, idx = tree.query(centers, k=2)
    pair_dists = dists[:, 1] / 2.0
    
    wall_active = np.where(walls < r + tol)[0]
    pair_active = np.where(pair_dists < r + tol)[0]
    
    return wall_active, pair_active, walls_x, walls_y, dists, idx

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

def _soft_min_obj_and_grad(x, tau=0.008):
    """Combined objective and gradient for L-BFGS efficiency."""
    centers = x.reshape(-1, 2)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    # Pairwise distances using cKDTree
    tree = cKDTree(centers)
    dists, idx = tree.query(centers, k=2)
    pair_dists = dists[:, 1] / 2.0
    
    # Combine all constraint distances
    all_dists = np.concatenate([walls, pair_dists])
    
    # Soft-min using log-sum-exp
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    exp_vals = np.exp(-shifted / tau)
    weights = exp_vals / np.sum(exp_vals)
    log_sum = np.log(np.sum(exp_vals))
    soft_min = min_d - tau * log_sum
    
    # Gradient
    grad = np.zeros_like(x)
    
    # Wall gradient contributions
    for i in range(N):
        w_wall = weights[i]
        if walls[i] == wall_dists_x[i]:
            grad[2*i] += w_wall * np.sign(centers[i, 0])
        if walls[i] == wall_dists_y[i]:
            grad[2*i+1] += w_wall * np.sign(centers[i, 1])
    
    # Pairwise gradient contributions (from nearest neighbors only)
    for k in range(N):
        w_pair = weights[N + k]
        i, j = k, idx[k, 1]
        d = dists[k, 1]
        if d > 1e-12:
            factor = 0.5 * w_pair / d
            diff_ij = centers[i] - centers[j]
            grad[2*i:2*i+2] -= factor * diff_ij
            grad[2*j:2*j+2] += factor * diff_ij
    
    return -soft_min, grad

def _grad_soft_min(x, tau=0.008):
    """
    Compute gradient of -soft_min objective using cKDTree for efficiency.
    """
    centers = x.reshape(-1, 2)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    # Pairwise distances using cKDTree (nearest neighbor only)
    tree = cKDTree(centers)
    dists, idx = tree.query(centers, k=2)
    pair_dists = dists[:, 1] / 2.0
    
    all_dists = np.concatenate([walls, pair_dists])
    
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
    
    # Pairwise gradient contributions (nearest neighbors only)
    for k in range(N):
        w_pair = weights[N + k]
        i, j = k, idx[k, 1]
        d = dists[k, 1]
        if d > 1e-12:
            factor = 0.5 * w_pair / d
            diff_ij = centers[i] - centers[j]
            grad[2*i:2*i+2] -= factor * diff_ij
            grad[2*j:2*j+2] += factor * diff_ij
    
    return grad

def _initial_grid():
    """Create initial grid layout with slight jitter."""
    # Use a hexagonal-like arrangement for better packing
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N].copy()
    
    rng = np.random.default_rng(12345)
    grid = grid + rng.normal(0, 0.005, grid.shape)
    grid = np.clip(grid, -0.49, 0.49)
    
    return grid

def _hex_grid():
    """Create hexagonal lattice initialization."""
    # For 50 circles, use a hexagonal arrangement
    # 7 columns: 4,5,4,5,4,5,3 = 30, need more
    # Try 8 columns with varying rows
    # Better: use a proper hex packing
    points = []
    # Hex lattice with spacing s
    s = 0.135
    rows = 7
    for row in range(rows):
        y = (row - 3) * s * np.sqrt(3) / 2
        if row % 2 == 0:
            x_start = -3 * s
            n_cols = 7
        else:
            x_start = -3 * s + s/2
            n_cols = 7
        for col in range(n_cols):
            x = x_start + col * s
            points.append([x, y])
    
    points = np.array(points[:N])
    # Center the arrangement
    cx, cy = points[:, 0].mean(), points[:, 1].mean()
    points[:, 0] -= cx
    points[:, 1] -= cy
    # Scale to fit
    max_extent = np.max(np.abs(points))
    if max_extent > 0.45:
        points *= 0.45 / max_extent
    
    rng = np.random.default_rng(42)
    points += rng.normal(0, 0.003, points.shape)
    points = np.clip(points, -0.49, 0.49)
    
    return points

def _simulated_annealing(centers, rng, time_budget):
    """
    Simulated annealing to optimize the minimum radius directly.
    Proposes random moves and accepts based on improvement in min radius.
    Uses incremental updates for speed.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    # Temperature schedule
    T_start = 0.015
    T_end = 0.00005
    n_steps = 20000
    cooling = (T_end / T_start) ** (1.0 / max(n_steps, 1))
    T = T_start
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        # Propose a random move
        idx = rng.integers(0, N)
        move = rng.normal(0, T, 2)
        trial = current.copy()
        trial[idx] += move
        trial[idx] = np.clip(trial[idx], -0.49, 0.49)
        
        trial_r = _compute_radius(trial)
        delta = trial_r - best_r
        
        # Accept if better or with probability exp(delta/T)
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
            current = trial
            if trial_r > best_r:
                best_r = trial_r
                best = trial.copy()
        
        T *= cooling
    
    return best, best_r

def _force_directed_relax(centers, rng, time_budget):
    """
    Force-directed relaxation to spread circles.
    Uses repulsive forces between circles and from walls.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    # Parameters
    k_repulse = 1.0
    k_wall = 1.0
    damping = 0.8
    momentum = np.zeros_like(current)
    step_size = 0.01
    
    for step in range(5000):
        if time.time() - start_time > time_budget:
            break
            
        forces = np.zeros_like(current)
        
        # Pairwise repulsive forces
        # O(N^2) is fine for N=50
        for i in range(N):
            for j in range(i + 1, N):
                diff = current[i] - current[j]
                dist = np.linalg.norm(diff)
                if dist > 1e-6:
                    # Force magnitude ~ 1/dist^2
                    mag = k_repulse / (dist * dist)
                    unit = diff / dist
                    forces[i] += mag * unit
                    forces[j] -= mag * unit
                else:
                    # Random jitter if overlapping
                    forces[i] += rng.normal(0, 0.1, 2)
                    forces[j] -= rng.normal(0, 0.1, 2)
        
        # Wall repulsive forces
        for i in range(N):
            # x-walls
            wx_left = current[i, 0] + 0.5
            wx_right = 0.5 - current[i, 0]
            if wx_left < 0.1:
                forces[i, 0] += k_wall / (wx_left * wx_left + 1e-6)
            if wx_right < 0.1:
                forces[i, 0] -= k_wall / (wx_right * wx_right + 1e-6)
            
            # y-walls
            wy_bottom = current[i, 1] + 0.5
            wy_top = 0.5 - current[i, 1]
            if wy_bottom < 0.1:
                forces[i, 1] += k_wall / (wy_bottom * wy_bottom + 1e-6)
            if wy_top < 0.1:
                forces[i, 1] -= k_wall / (wy_top * wy_top + 1e-6)
                
        # Update positions
        momentum = damping * momentum + forces
        current += step_size * momentum
        current = np.clip(current, -0.5, 0.5)
        
        # Anneal step size
        step_size *= 0.995
        
        r_curr = _compute_radius(current)
        if r_curr > best_r:
            best_r = r_curr
            best = current.copy()
            
    return best, best_r

def _lbfgs_optimize(x0, tau=0.008, time_budget=20.0):
    """Run L-BFGS-B optimization on the soft-min objective with combined obj/grad."""
    start_time = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        return _soft_min_obj_and_grad(x, tau)
    
    try:
        result = opt.minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12,
                     'maxfun': 2000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)

def _high_precision_polish(centers, time_budget):
    """
    High-precision polish using progressive tau reduction.
    Multiple passes with decreasing tau for tighter convergence.
    """
    start_time = time.time()
    current = centers.copy()
    
    # Progressive tau reduction: each pass sharpens the min approximation
    taus = [0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    for tau in taus:
        if time.time() - start_time > time_budget:
            break
        
        x0 = current.flatten().copy()
        
        def obj_and_grad(x, _tau=tau):
            if time.time() - start_time > time_budget:
                return 0.0, np.zeros_like(x)
            return _soft_min_obj_and_grad(x, _tau)
        
        try:
            result = opt.minimize(
                obj_and_grad, x0, jac=True, method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13,
                         'maxfun': 1000, 'disp': False}
            )
            polished = result.x.reshape(-1, 2)
            polished = np.clip(polished, -0.5, 0.5)
            
            r_new = _compute_radius(polished)
            r_old = _compute_radius(current)
            if r_new >= r_old - 1e-12:
                current = polished
        except Exception:
            pass
    
    return current

def _exact_local_solve(centers, time_budget):
    """
    Exact local solve: identify active constraints and solve the 
    equality-constrained problem exactly.
    """
    start_time = time.time()
    current = centers.copy()
    r = _compute_radius(current)
    
    # Try to improve by solving the active constraint system
    for iteration in range(20):
        if time.time() - start_time > time_budget:
            break
        
        wall_active, pair_active, walls_x, walls_y, dists, idx = _active_constraints(
            current, r, tol=1e-8
        )
        
        # Build the system of active constraints
        # For wall constraints: center[i, dim] = ±(0.5 - r)
        # For pair constraints: |center[i] - center[j]| = 2r
        
        # Use a simple iterative approach: adjust points to satisfy active constraints
        improved = False
        
        # Adjust wall-constrained points
        for i in wall_active:
            if walls_x[i] < r + 1e-8:
                # x is limiting
                current[i, 0] = np.sign(current[i, 0]) * (0.5 - r)
                improved = True
            if walls_y[i] < r + 1e-8:
                # y is limiting
                current[i, 1] = np.sign(current[i, 1]) * (0.5 - r)
                improved = True
        
        # Adjust pair-constrained points (move apart)
        for k in pair_active:
            i, j = k, idx[k, 1]
            d = dists[k, 1]
            if d < 2 * r - 1e-8 and d > 1e-12:
                # Need to push apart
                diff = current[i] - current[j]
                unit = diff / d
                needed = 2 * r - d
                current[i] += 0.5 * needed * unit
                current[j] -= 0.5 * needed * unit
                improved = True
        
        current = np.clip(current, -0.5, 0.5)
        
        r_new = _compute_radius(current)
        if r_new > r:
            r = r_new
        elif not improved:
            break
    
    return current

def _binary_search_radius(centers, lo, hi, time_budget):
    """Binary search for maximum feasible radius with soft-min polishing."""
    start_time = time.time()
    best = centers.copy()
    best_r = lo
    
    for _ in range(15):
        if time.time() - start_time > time_budget:
            break
        
        mid = (lo + hi) / 2
        trial = best.copy()
        
        # Use L-BFGS with soft-min to try to achieve radius mid
        x = trial.flatten().copy()
        
        # Objective: penalize constraints that are below mid
        def violation_obj(x):
            c = x.reshape(-1, 2)
            walls_x = 0.5 - np.abs(c[:, 0])
            walls_y = 0.5 - np.abs(c[:, 1])
            walls = np.minimum(walls_x, walls_y)
            
            tree = cKDTree(c)
            dists, _ = tree.query(c, k=2)
            pair_dists = dists[:, 1] / 2.0
            
            # Soft violation: sum of exp violations
            wall_viol = np.sum(np.maximum(0, mid - walls))
            pair_viol = np.sum(np.maximum(0, mid - pair_dists))
            return wall_viol + pair_viol
        
        try:
            result = opt.minimize(
                violation_obj, x, method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10,
                         'maxfun': 300, 'disp': False}
            )
            trial = result.x.reshape(-1, 2)
            trial = np.clip(trial, -0.5, 0.5)
        except Exception:
            pass
        
        trial_r = _compute_radius(trial)
        
        if trial_r >= mid - 1e-9:
            lo = mid
            best = trial
            best_r = trial_r
        else:
            hi = mid
    
    return best, best_r

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Initial configuration: try both grid and hex
    best_centers = _hex_grid()
    best_radius = _compute_radius(best_centers)
    
    grid_centers = _initial_grid()
    grid_radius = _compute_radius(grid_centers)
    if grid_radius > best_radius:
        best_centers = grid_centers
        best_radius = grid_radius
    
    rng = np.random.default_rng(777)
    
    # Phase 1: Force-Directed Relaxation (shorter)
    fd_budget = min(15.0, (deadline - time.time()) * 0.2)
    if fd_budget > 3:
        trial_centers, trial_r = _force_directed_relax(best_centers, rng, fd_budget)
        if trial_r > best_radius:
            best_radius = trial_r
            best_centers = trial_centers
    
    # Phase 2: Simulated Annealing for global exploration (shorter)
    sa_budget = min(12.0, (deadline - time.time()) * 0.15)
    if sa_budget > 3:
        trial_centers, trial_r = _simulated_annealing(best_centers, rng, sa_budget)
        if trial_r > best_radius:
            best_radius = trial_r
            best_centers = trial_centers
    
    # Phase 3: L-BFGS optimization with soft-min (primary workhorse)
    # Progressive tau: start coarse, end fine
    x0 = best_centers.flatten()
    for tau in [0.02, 0.01, 0.005, 0.002]:
        remaining = deadline - time.time()
        if remaining < 15:
            break
        tau_budget = min(15.0, remaining * 0.35)
        if tau_budget > 3:
            trial = _lbfgs_optimize(x0, tau=tau, time_budget=tau_budget)
            trial_r = _compute_radius(trial)
            if trial_r > best_radius:
                best_radius = trial_r
                best_centers = trial
            x0 = best_centers.flatten()
    
    # Phase 4: Binary search with soft-min polish
    if time.time() < deadline - 20:
        lo = best_radius
        hi = min(0.08, best_radius * 1.03)
        bs_budget = min(10.0, deadline - time.time() - 10)
        if bs_budget > 3:
            best_centers, best_radius = _binary_search_radius(
                best_centers, lo, hi, time_budget=bs_budget
            )
    
    # Phase 5: High-precision polish with progressive tau
    if time.time() < deadline - 5:
        hp_budget = min(15.0, deadline - time.time() - 2)
        if hp_budget > 3:
            trial = _high_precision_polish(best_centers, time_budget=hp_budget)
            trial_r = _compute_radius(trial)
            if trial_r > best_radius:
                best_radius = trial_r
                best_centers = trial
    
    # Phase 6: Exact local solve for final tightening
    if time.time() < deadline - 2:
        el_budget = min(5.0, deadline - time.time() - 1)
        if el_budget > 1:
            trial = _exact_local_solve(best_centers, time_budget=el_budget)
            trial_r = _compute_radius(trial)
            if trial_r > best_radius:
                best_radius = trial_r
                best_centers = trial
    
    # Final validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_grid()
    
    return best_centers
