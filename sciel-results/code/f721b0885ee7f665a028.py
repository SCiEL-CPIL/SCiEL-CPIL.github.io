"""Pack 50 equal circles in a unit square using L-BFGS-B optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
HALF = 0.5

def _compute_radius(centers):
    """Compute the largest valid radius for the given centers."""
    # Distance between centers
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    # Distance to walls
    wall_dists = np.minimum(
        np.minimum(HALF - centers[:, 0], centers[:, 0] + HALF),
        np.minimum(HALF - centers[:, 1], centers[:, 1] + HALF)
    )
    min_wall = np.min(wall_dists)
    
    return min(min_dist / 2.0, min_wall)

def _objective(centers_flat, r_target):
    """
    Objective function: sum of squared violations of constraints.
    We want all pairwise distances >= 2*r_target and all centers within
    [-(HALF - r_target), HALF - r_target] in each coordinate.
    """
    centers = centers_flat.reshape(N, 2)
    
    # Pairwise distance violations
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # For each pair, if dist < 2*r_target, we have a violation
    target_dist = 2.0 * r_target
    violations = np.maximum(0.0, target_dist - dists)
    pair_violation = np.sum(violations ** 2)
    
    # Wall violations: center must be in [-(HALF - r), HALF - r]
    boundary = HALF - r_target
    # x violations
    x_viol = np.maximum(0.0, centers[:, 0] - boundary) + np.maximum(0.0, -boundary - centers[:, 0])
    # y violations
    y_viol = np.maximum(0.0, centers[:, 1] - boundary) + np.maximum(0.0, -boundary - centers[:, 1])
    wall_violation = np.sum(x_viol ** 2) + np.sum(y_viol ** 2)
    
    return pair_violation + wall_violation

def _grid_init(n_cols=None):
    """Initialize with a grid layout, possibly with some jitter."""
    if n_cols is None:
        n_cols = 7
    n_rows = int(np.ceil(N / n_cols))
    
    # Create grid positions
    xs = np.linspace(-HALF + 0.08, HALF - 0.08, n_cols)
    ys = np.linspace(-HALF + 0.08, HALF - 0.08, n_rows)
    
    centers = []
    for i in range(N):
        row = i // n_cols
        col = i % n_cols
        centers.append([xs[col], ys[row]])
    
    return np.array(centers)

def _random_init(seed=None):
    """Random initialization within the square."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    margin = 0.05
    centers = rng.uniform(-HALF + margin, HALF - margin, size=(N, 2))
    return centers

def _optimize_from(centers, r_target, time_limit=10.0):
    """Run L-BFGS-B optimization from a starting configuration."""
    start_time = time.time()
    x0 = centers.flatten()
    
    bounds = [(-HALF + r_target, HALF - r_target)] * (2 * N)
    
    result = minimize(
        _objective,
        x0,
        args=(r_target,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
    )
    
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        # Return best found so far
        best_x = result.x if result.x is not None else x0
        return best_x.reshape(N, 2), result.fun if result.fun is not None else _objective(x0, r_target)
    
    return result.x.reshape(N, 2), result.fun

def _binary_search_radius(centers, r_low, r_high, max_iter=30, time_budget=60.0):
    """Binary search for the maximum radius."""
    start_time = time.time()
    best_centers = centers.copy()
    best_radius = r_low
    
    for _ in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        centers_opt, obj_val = _optimize_from(centers, r_mid, time_limit=15.0)
        
        # Check if feasible
        if obj_val < 1e-8:
            # Feasible, try larger radius
            best_centers = centers_opt.copy()
            best_radius = r_mid
            r_low = r_mid
        else:
            r_high = r_mid
        
        if r_high - r_low < 1e-9:
            break
    
    return best_centers, best_radius

def construct():
    """Construct the best packing of 50 circles in a unit square."""
    global_start = time.time()
    deadline = global_start + 120.0  # Leave margin before 150s limit
    
    # Start with grid initialization
    centers = _grid_init(7)
    
    # Initial radius estimate
    r_low = 0.05
    r_high = 0.10
    
    # First pass: binary search with grid init
    centers, radius = _binary_search_radius(centers, r_low, r_high, max_iter=20, time_budget=40.0)
    
    # Check time
    if time.time() > deadline - 30:
        return centers
    
    # Second pass: try different initializations
    best_radius = radius
    best_centers = centers.copy()
    
    # Try random inits
    for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]:
        if time.time() > deadline - 15:
            break
        
        centers_rand = _random_init(seed)
        # Narrow the search range around current best
        r_low_narrow = max(r_low, best_radius - 0.01)
        r_high_narrow = min(r_high, best_radius + 0.01)
        
        if r_high_narrow <= r_low_narrow:
            r_low_narrow = best_radius * 0.95
            r_high_narrow = best_radius * 1.05
        
        centers_opt, r_opt = _binary_search_radius(
            centers_rand, r_low_narrow, r_high_narrow, max_iter=10, time_budget=10.0
        )
        
        if r_opt > best_radius:
            best_radius = r_opt
            best_centers = centers_opt.copy()
    
    # Final refinement from best center
    if time.time() < deadline - 5:
        r_low_final = best_radius * 0.99
        r_high_final = best_radius * 1.01
        centers_final, r_final = _binary_search_radius(
            best_centers, r_low_final, r_high_final, max_iter=15, time_budget=15.0
        )
        if r_final > best_radius:
            best_radius = r_final
            best_centers = centers_final
    
    return best_centers
