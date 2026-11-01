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

def _penalty_objective(x, r_target, penalty_weight=1.0):
    """
    Objective: sum of squared violations for target radius r_target.
    x: flattened array of 2*N coordinates
    """
    centers = x.reshape(-1, 2)
    
    # Wall violations
    wall_x = 0.5 - np.abs(centers[:, 0])
    wall_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_x, wall_y)
    wall_viol = np.maximum(0, r_target - walls)
    
    # Pair violations using cKDTree
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    pair_dists = dists[:, 1]
    pair_viol = np.maximum(0, 2.0 * r_target - pair_dists)
    
    # Penalty: sum of squared violations
    pen = np.sum(wall_viol**2) + np.sum(pair_viol**2)
    
    return penalty_weight * pen

def _grad_penalty(x, r_target, penalty_weight=1.0):
    """
    Analytical gradient of the penalty objective.
    """
    centers = x.reshape(-1, 2)
    grad = np.zeros_like(x)
    
    # Wall gradient contributions
    wall_x = 0.5 - np.abs(centers[:, 0])
    wall_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_x, wall_y)
    wall_viol = np.maximum(0, r_target - walls)
    
    for i in range(N):
        if wall_viol[i] > 0:
            w_wall = 2.0 * penalty_weight * wall_viol[i]
            if walls[i] <= wall_x[i] + 1e-15:
                grad[2*i] += w_wall * np.sign(centers[i, 0])
            if walls[i] <= wall_y[i] + 1e-15:
                grad[2*i+1] += w_wall * np.sign(centers[i, 1])
    
    # Pair gradient contributions
    tree = cKDTree(centers)
    dists, idx = tree.query(centers, k=2)
    pair_dists = dists[:, 1]
    pair_viol = np.maximum(0, 2.0 * r_target - pair_dists)
    
    for i in range(N):
        if pair_viol[i] > 0:
            j = idx[i, 1]
            d = pair_dists[i]
            if d > 1e-12:
                factor = 2.0 * penalty_weight * pair_viol[i] / d
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

def _simulated_annealing(centers, rng, time_budget):
    """
    Simulated annealing to optimize the minimum radius directly.
    Uses faster evaluation and adaptive move sizes.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    # Temperature schedule
    T_start = 0.015
    T_end = 0.00005
    n_steps = 30000
    cooling = (T_end / T_start) ** (1.0 / max(n_steps, 1))
    T = T_start
    
    # Precompute initial tree
    tree = cKDTree(current)
    dists, idx_nn = tree.query(current, k=2)
    wall_x = 0.5 - np.abs(current[:, 0])
    wall_y = 0.5 - np.abs(current[:, 1])
    min_pair = np.min(dists[:, 1]) / 2.0
    min_wall = np.min(np.minimum(wall_x, wall_y))
    curr_r = min(min_pair, min_wall)
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        # Weighted selection of constraint to improve
        all_dists = np.concatenate([np.minimum(wall_x, wall_y), dists[:, 1] / 2.0])
        min_d = np.min(all_dists)
        weights = np.exp(-(all_dists - min_d) / 0.004)
        weights /= np.sum(weights)
        
        idx = rng.choice(len(all_dists), p=weights)
        
        move_size = 0.003 + 0.008 * (T / T_start)
        
        if idx < N:
            # Wall constraint for circle idx
            i = idx
            # Move away from the limiting wall
            if wall_x[i] <= wall_y[i]:
                delta = np.array([np.sign(current[i, 0]) if current[i, 0] != 0 else rng.choice([-1, 1]), 0.0])
            else:
                delta = np.array([0.0, np.sign(current[i, 1]) if current[i, 1] != 0 else rng.choice([-1, 1])])
            # Add some randomness
            delta += rng.normal(0, 0.002, 2)
            trial = current.copy()
            trial[i] += delta * move_size
            trial = np.clip(trial, -0.5, 0.5)
        else:
            # Pair constraint - move apart from nearest neighbor
            i = idx - N
            j = idx_nn[i, 1]
            
            diff = current[i] - current[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-9:
                unit = diff / dist
            else:
                unit = rng.normal(size=2)
                unit /= np.linalg.norm(unit)
            
            # Move both apart with some randomness
            trial = current.copy()
            trial[i] += unit * move_size * rng.uniform(0.3, 0.8)
            trial[j] -= unit * move_size * rng.uniform(0.3, 0.8)
            # Add small random perturbation
            trial[i] += rng.normal(0, 0.001, 2)
            trial[j] += rng.normal(0, 0.001, 2)
            trial = np.clip(trial, -0.5, 0.5)
        
        # Fast evaluation
        trial_tree = cKDTree(trial)
        trial_dists, _ = trial_tree.query(trial, k=2)
        trial_min_pair = np.min(trial_dists[:, 1]) / 2.0
        trial_wall_x = 0.5 - np.abs(trial[:, 0])
        trial_wall_y = 0.5 - np.abs(trial[:, 1])
        trial_min_wall = np.min(np.minimum(trial_wall_x, trial_wall_y))
        trial_r = min(trial_min_pair, trial_min_wall)
        
        delta_r = trial_r - curr_r
        
        # Accept if improvement, or with probability exp(delta_r / T)
        if delta_r >= 0:
            current = trial
            curr_r = trial_r
            # Update cached values
            tree = trial_tree
            dists = trial_dists
            idx_nn = trial_tree.query(trial, k=2)[1]
            wall_x = trial_wall_x
            wall_y = trial_wall_y
            min_pair = trial_min_pair
            min_wall = trial_min_wall
            if trial_r > best_r:
                best_r = trial_r
                best = trial.copy()
        else:
            if rng.random() < np.exp(delta_r / max(T, 1e-10)):
                current = trial
                curr_r = trial_r
                tree = trial_tree
                dists = trial_dists
                idx_nn = trial_tree.query(trial, k=2)[1]
                wall_x = trial_wall_x
                wall_y = trial_wall_y
                min_pair = trial_min_pair
                min_wall = trial_min_wall
        
        T *= cooling
    
    return best, best_r

def _lbfgs_penalty_optimize(x0, r_target, time_budget, penalty_weight=1.0):
    """
    Run L-BFGS-B optimization on the penalty objective for a target radius.
    """
    start_time = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        f = _penalty_objective(x, r_target, penalty_weight)
        g = _grad_penalty(x, r_target, penalty_weight)
        return f, g
    
    try:
        result = opt.minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12,
                     'maxfun': 2000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)

def _binary_search_radius(centers, lo, hi, time_budget):
    """
    Binary search for maximum feasible radius with penalty-based L-BFGS.
    """
    start_time = time.time()
    best = centers.copy()
    best_r = lo
    
    for _ in range(15):  # More iterations for finer binary search
        if time.time() - start_time > time_budget:
            break
        
        mid = (lo + hi) / 2
        x0 = best.flatten().copy()
        
        # Use L-BFGS to minimize violation for radius mid
        trial = _lbfgs_penalty_optimize(x0, mid, time_budget=min(8, deadline - time.time() - 5))
        trial_r = _compute_radius(trial)
        
        if trial_r >= mid - 1e-10:
            lo = mid
            if trial_r > best_r:
                best_r = trial_r
                best = trial
        else:
            hi = mid
    
    return best, best_r

def _high_precision_polish(centers, time_budget):
    """
    High-precision polish using binary search with very tight tolerances.
    """
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)
    
    # Do multiple passes of binary search with decreasing ranges
    for pass_num in range(3):
        if time.time() - start_time > time_budget:
            break
        
        # Start with a range slightly above current best
        lo = best_r
        hi = min(0.08, best_r * 1.01)
        
        best, best_r = _binary_search_radius(
            best, lo, hi, 
            time_budget=min(15, deadline - time.time() - 10)
        )
    
    return best, best_r

# Global RNG for use
rng_global = np.random.default_rng(42)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Initial configuration
    best_centers = _initial_grid()
    best_radius = _compute_radius(best_centers)
    
    rng = np.random.default_rng(777)
    
    # Phase 1: Simulated Annealing to escape local minima
    sa_budget = min(60.0, (deadline - time.time()) * 0.5)
    if sa_budget > 5:
        best_centers, best_radius = _simulated_annealing(best_centers, rng, sa_budget)
    
    # Phase 2: Binary search with penalty-based L-BFGS
    if time.time() < deadline - 40:
        lo = best_radius
        hi = min(0.08, best_radius * 1.02)
        best_centers, best_radius = _binary_search_radius(
            best_centers, lo, hi,
            time_budget=min(30, deadline - time.time() - 20)
        )
    
    # Phase 3: High-precision polish
    if time.time() < deadline - 20:
        best_centers, best_radius = _high_precision_polish(
            best_centers, time_budget=min(20, deadline - time.time() - 10)
        )
    
    # Final validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_grid()
    
    return best_centers
