"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    if len(centers) < 2:
        min_dist_to_wall = np.min(0.5 - np.abs(centers))
        return min_dist_to_wall
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    min_wall_dist = np.min(0.5 - np.abs(centers))
    return min(min_pair_dist / 2.0, min_wall_dist)

def _energy_and_grad(centers, target_r):
    """Compute energy and gradient for repulsion model."""
    n = len(centers)
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # Pairwise repulsion
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Mask for pairs closer than 2*target_r
    mask = dists < 2.0 * target_r
    violations = np.where(mask, 2.0 * target_r - dists, 0.0)
    energy += np.sum(violations**2)
    
    # Vectorized gradient contribution from pairs
    unit_vecs = diff / dists[:, :, np.newaxis]
    coeff = np.where(mask, -2.0 * (2.0 * target_r - dists), 0.0)
    pair_grad = np.sum(coeff[:, :, np.newaxis] * unit_vecs, axis=1)
    grad += pair_grad
    
    # Wall repulsion
    dw = np.zeros((n, 6))
    dw[:, 0] = 0.5 - centers[:, 0]
    dw[:, 1] = 0.5 + centers[:, 0]
    dw[:, 2] = 0.5 - centers[:, 1]
    dw[:, 3] = 0.5 + centers[:, 1]
    dw[:, 4] = 0.5 - centers[:, 2]
    dw[:, 5] = 0.5 + centers[:, 2]
    
    min_wall_dist = np.min(dw, axis=1)
    wall_violations = np.maximum(0, target_r - min_wall_dist)
    energy += np.sum(wall_violations**2)
    
    min_idx = np.argmin(dw, axis=1)
    wall_grad = np.zeros_like(centers)
    violated = wall_violations > 0
    
    for i in range(n):
        if violated[i]:
            idx = min_idx[i]
            factor = -2.0 * (target_r - min_wall_dist[i])
            if idx == 0: wall_grad[i, 0] += -factor * (-1)
            elif idx == 1: wall_grad[i, 0] += -factor * (1)
            elif idx == 2: wall_grad[i, 1] += -factor * (-1)
            elif idx == 3: wall_grad[i, 1] += -factor * (1)
            elif idx == 4: wall_grad[i, 2] += -factor * (-1)
            elif idx == 5: wall_grad[i, 2] += -factor * (1)
            
    grad += wall_grad
    
    return energy, grad

def _optimize_from_init(centers_init, target_r, max_iter=500, tol=1e-12):
    """Optimize from a given initial configuration using L-BFGS-B."""
    def objective(x):
        c = x.reshape(-1, 3)
        e, g = _energy_and_grad(c, target_r)
        return e, g
    
    x0 = centers_init.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8, 'disp': False}
    )
    
    return result.x.reshape(-1, 3), result.success, result.fun

def _stratified_grid_init(n, n_layers=4):
    """Generate a stratified grid with more uniform distribution."""
    # Try multiple grid configurations and pick the best
    best_grid = None
    best_r = 0.0
    
    for m in range(4, 8):
        m = int(np.ceil(n ** (1.0/3.0)))
        if m < 4:
            m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        if len(grid) >= n:
            grid = grid[:n]
            r = _compute_radius(grid)
            if r > best_r:
                best_r = r
                best_grid = grid.copy()
    
    if best_grid is None:
        m = int(np.ceil(n ** (1.0/3.0)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_grid = grid[:n]
    
    return best_grid

def _random_init(n, seed=None):
    """Generate random initial configuration inside the cube."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))

def _binary_search_radius(centers, r_low, r_high, max_iter=15, time_limit=60.0):
    """Binary search for the maximum achievable radius."""
    start = time.time()
    best_centers = centers.copy()
    best_radius = r_low
    
    for i in range(max_iter):
        if time.time() - start > time_limit:
            break
        
        r_mid = (r_low + r_high) / 2.0
        try:
            optimized, success, energy = _optimize_from_init(
                best_centers, r_mid, max_iter=100, tol=1e-10
            )
            r_opt = _compute_radius(optimized)
            
            if r_opt >= r_mid * 0.999:
                # Feasible, try larger radius
                r_low = r_mid
                best_radius = r_opt
                best_centers = optimized.copy()
            else:
                # Infeasible, try smaller radius
                r_high = r_mid
        except Exception:
            r_high = r_mid
    
    return best_centers, best_radius

def _polish_solution(centers, radius, time_limit=20.0):
    """High-precision polish pass with tighter tolerances."""
    start = time.time()
    best_centers = centers.copy()
    best_radius = radius
    
    # Try increasing radius slightly and re-optimizing with tight tolerance
    for attempt in range(3):
        if time.time() - start > time_limit:
            break
        
        r_target = best_radius * (1.001 + attempt * 0.0005)
        try:
            optimized, success, energy = _optimize_from_init(
                best_centers, r_target, max_iter=300, tol=1e-14
            )
            r_opt = _compute_radius(optimized)
            if r_opt > best_radius:
                best_radius = r_opt
                best_centers = optimized.copy()
        except Exception:
            break
    
    # Final local refinement with very tight tolerance
    if time.time() - start < time_limit - 5:
        try:
            optimized, success, energy = _optimize_from_init(
                best_centers, best_radius, max_iter=500, tol=1e-15
            )
            r_opt = _compute_radius(optimized)
            if r_opt > best_radius:
                best_radius = r_opt
                best_centers = optimized.copy()
        except Exception:
            pass
    
    return best_centers, best_radius

def construct():
    """Construct the optimal packing of N spheres in the unit cube."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # Strategy 1: Stratified grid initialization
    init_grid = _stratified_grid_init(N)
    r_grid = _compute_radius(init_grid)
    best_radius = r_grid
    best_centers = init_grid.copy()
    
    # Strategy 2: Initial optimization from grid
    target_r = best_radius * 1.02
    try:
        optimized, success, energy = _optimize_from_init(
            best_centers, target_r, max_iter=300, tol=1e-10
        )
        r_opt = _compute_radius(optimized)
        if r_opt > best_radius:
            best_radius = r_opt
            best_centers = optimized.copy()
    except Exception:
        pass
    
    # Strategy 3: Binary search for maximum radius
    if time.time() < deadline - 40:
        r_low = best_radius
        r_high = best_radius * 1.15  # Upper bound estimate
        
        # First check if r_high is feasible
        try:
            test_opt, _, _ = _optimize_from_init(
                best_centers, r_high, max_iter=50, tol=1e-8
            )
            r_test = _compute_radius(test_opt)
            if r_test < r_high * 0.999:
                r_high = (r_low + r_high) / 2.0
        except Exception:
            pass
        
        best_centers, best_radius = _binary_search_radius(
            best_centers, r_low, r_high, max_iter=12, 
            time_limit=min(40.0, deadline - time.time() - 25)
        )
    
    # Strategy 4: Try a few random restarts if time allows
    for seed in range(3):
        if time.time() > deadline - 30:
            break
        
        init_rand = _random_init(N, seed=seed + 100)
        r_rand = _compute_radius(init_rand)
        
        # Only proceed if random start is competitive
        if r_rand > best_radius * 0.8:
            try:
                target_r = max(r_rand, best_radius * 0.95)
                optimized, success, energy = _optimize_from_init(
                    init_rand, target_r, max_iter=200, tol=1e-10
                )
                r_opt = _compute_radius(optimized)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = optimized.copy()
            except Exception:
                continue
    
    # Strategy 5: Final high-precision polish
    if time.time() < deadline - 10 and best_centers is not None:
        best_centers, best_radius = _polish_solution(
            best_centers, best_radius, 
            time_limit=min(20.0, deadline - time.time() - 5)
        )
    
    # Final check and clip
    if best_centers is None:
        best_centers = _stratified_grid_init(N)
        
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    # Verify validity
    final_r = _compute_radius(best_centers)
    if not np.isfinite(final_r) or final_r <= 0:
        best_centers = _stratified_grid_init(N)
        best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
# EVOLVE-BLOCK-END
