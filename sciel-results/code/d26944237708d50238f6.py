"""Combined sphere packing: lattice-based greedy init + soft-min L-BFGS-B optimization."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay, cKDTree

N = 100
CUBE_LIMIT = 0.5

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    if len(centers) < 2:
        return float(np.min(CUBE_LIMIT - np.abs(centers)))
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    min_wall_dist = np.min(CUBE_LIMIT - np.abs(centers))
    
    return float(min(min_pair_dist / 2.0, min_wall_dist))

def _softmin_objective_and_grad(x, beta):
    """
    Objective: minimize negative soft-min of distances (pairwise and wall).
    x: flattened centers (N*3,)
    beta: inverse temperature (larger beta = sharper min)
    """
    centers = x.reshape(N, 3)
    
    # Pairwise distances squared
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Wall distances
    wall_dists = CUBE_LIMIT - np.abs(centers)
    wall_sq_dists = wall_dists**2
    
    # Soft-min on squared distances for pairs
    finite_mask = np.isfinite(sq_dists)
    if np.any(finite_mask):
        max_sq = np.max(sq_dists[finite_mask])
        shifted = sq_dists - max_sq
        exp_terms = np.exp(-beta * shifted)
        exp_terms[~finite_mask] = 0.0
        sum_exp = np.sum(exp_terms)
        
        if sum_exp > 0:
            log_sum = np.log(sum_exp)
            soft_min_sq = max_sq - (1.0 / beta) * log_sum
        else:
            soft_min_sq = max_sq
    else:
        soft_min_sq = 0.0
        
    # Wall part
    max_wall_sq = np.max(wall_sq_dists)
    shifted_wall = wall_sq_dists - max_wall_sq
    exp_wall = np.exp(-beta * shifted_wall)
    sum_wall = np.sum(exp_wall)
    
    if sum_wall > 0:
        log_sum_wall = np.log(sum_wall)
        soft_min_wall_sq = max_wall_sq - (1.0 / beta) * log_sum_wall
    else:
        soft_min_wall_sq = max_wall_sq
        
    # Combine pair and wall soft-mins using another soft-min
    vals = np.array([soft_min_sq, soft_min_wall_sq])
    max_val = np.max(vals)
    exp_vals = np.exp(-beta * (vals - max_val))
    sum_vals = np.sum(exp_vals)
    
    if sum_vals > 0:
        log_sum_vals = np.log(sum_vals)
        total_soft_min_sq = max_val - (1.0 / beta) * log_sum_vals
    else:
        total_soft_min_sq = max_val
        
    total_soft_min_sq = max(total_soft_min_sq, 0.0)
    obj_val = np.sqrt(total_soft_min_sq)
    
    # Gradient
    grad = np.zeros_like(centers)
    
    if np.any(finite_mask) and sum_exp > 0:
        weights = exp_terms / sum_exp
        sum_w = np.sum(weights, axis=1)
        weighted_sum_x = weights @ centers
        
        grad_pair_sq = sum_w[:, np.newaxis] * centers - weighted_sum_x
        
        signs = np.sign(centers)
        signs[centers == 0] = 1.0
        d_wall_sq_dc = -2.0 * wall_dists * signs
        
        weights_wall = exp_wall / sum_wall
        grad_wall_sq = np.zeros_like(centers)
        for i in range(N):
            w = weights_wall[i]
            if w > 0:
                grad_wall_sq[i] += w * d_wall_sq_dc[i]
        grad_wall_sq *= -1.0 / beta
        
        w_A = np.exp(-beta * (soft_min_sq - max_val)) / sum_vals
        w_B = np.exp(-beta * (soft_min_wall_sq - max_val)) / sum_vals
        
        combined_grad_sq = w_A * grad_pair_sq + w_B * grad_wall_sq
        
        if total_soft_min_sq > 1e-12:
            grad = combined_grad_sq / (2.0 * obj_val)
        else:
            grad = np.zeros_like(centers)
            
    return -obj_val, -grad

def _optimize_packing(init_centers, time_limit=10.0):
    """
    Optimize packing from initial centers using soft-min L-BFGS-B.
    """
    start_time = time.time()
    x0 = init_centers.flatten()
    
    betas = [10, 50, 100, 500, 1000, 2000, 5000]
    
    best_x = x0
    best_val = _compute_radius(init_centers)
    
    for beta in betas:
        if time.time() - start_time > time_limit:
            break
            
        def obj_grad(x):
            if time.time() - start_time > time_limit:
                return 1e10, np.zeros_like(x)
            val, grad = _softmin_objective_and_grad(x, beta)
            if not np.all(np.isfinite(grad)):
                return 1e10, np.zeros_like(x)
            return val, grad
            
        try:
            result = minimize(
                obj_grad, x0,
                method='L-BFGS-B',
                bounds=[(-CUBE_LIMIT, CUBE_LIMIT)] * (N * 3),
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            x0 = result.x
            r = _compute_radius(x0.reshape(N, 3))
            if r > best_val:
                best_val = r
                best_x = x0
        except Exception:
            pass
            
    return best_x.reshape(N, 3)

def _lattice_greedy_init(n_points, lattice_size, target_radius, seed=None):
    """
    Greedy selection of n_points positions from a dense lattice
    such that selected points are at least 2*target_radius apart.
    """
    rng = np.random.RandomState(seed)
    
    # Create dense lattice
    ticks = np.linspace(-CUBE_LIMIT + target_radius, CUBE_LIMIT - target_radius, lattice_size)
    lattice = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Shuffle lattice points for diversity
    indices = rng.permutation(len(lattice))
    lattice_shuffled = lattice[indices]
    
    selected = []
    min_dist = 2.0 * target_radius
    
    for point in lattice_shuffled:
        if len(selected) >= n_points:
            break
        if selected:
            dists = np.linalg.norm(point - np.array(selected), axis=1)
            if np.min(dists) >= min_dist:
                selected.append(point)
    
    # If we didn't select enough, fill remaining with random points
    if len(selected) < n_points:
        remaining = n_points - len(selected)
        while len(selected) < n_points:
            p = rng.uniform(-CUBE_LIMIT + target_radius, CUBE_LIMIT - target_radius, size=3)
            if selected:
                dists = np.linalg.norm(p - np.array(selected), axis=1)
                if np.min(dists) >= min_dist * 0.95:  # slightly relaxed
                    selected.append(p)
            else:
                selected.append(p)
    
    return np.array(selected[:n_points])

def _grid_init(n):
    """Generate grid initial configuration."""
    m = int(np.ceil(n ** (1.0/3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]

def _jitter_init(n, base, amount=0.02, seed=None):
    """Add jitter to a base configuration."""
    rng = np.random.RandomState(seed)
    return np.clip(base + rng.uniform(-amount, amount, size=(n, 3)), -0.49, 0.49)

def _random_init(n, seed=None):
    """Generate random initial configuration."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))

def _final_polish(centers, time_limit=10.0):
    """High-precision final polish with very tight tolerances."""
    start_time = time.time()
    x0 = centers.flatten()
    
    # Use high beta for sharp minimum
    for beta in [5000, 10000, 20000]:
        if time.time() - start_time > time_limit:
            break
            
        def obj_grad(x):
            if time.time() - start_time > time_limit:
                return 1e10, np.zeros_like(x)
            val, grad = _softmin_objective_and_grad(x, beta)
            if not np.all(np.isfinite(grad)):
                return 1e10, np.zeros_like(x)
            return val, grad
            
        try:
            result = minimize(
                obj_grad, x0,
                method='L-BFGS-B',
                bounds=[(-CUBE_LIMIT, CUBE_LIMIT)] * (N * 3),
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
            )
            x0 = result.x
        except Exception:
            pass
            
    return x0.reshape(N, 3)

def construct():
    """Construct the optimal packing of N spheres in the unit cube."""
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s limit
    
    best_centers = None
    best_radius = 0.0
    
    # Generate diverse initializations
    inits = []
    
    # 1. Grid init
    grid = _grid_init(N)
    inits.append(('grid', grid))
    
    # 2. Lattice-based greedy inits at various target radii
    target_radii = [0.08, 0.09, 0.10, 0.105, 0.108, 0.110, 0.111]
    lattice_sizes = [12, 15, 18]
    
    for tr in target_radii:
        for ls in lattice_sizes:
            for seed in [0, 1, 2]:
                try:
                    lat_init = _lattice_greedy_init(N, ls, tr, seed=seed)
                    if len(lat_init) == N and np.all(np.isfinite(lat_init)):
                        inits.append((f'lat_{tr}_{ls}_{seed}', lat_init))
                except Exception:
                    continue
    
    # 3. Jittered grids
    for seed in range(3):
        inits.append((f'jitter_grid_{seed}', _jitter_init(N, grid, amount=0.02, seed=seed)))
    
    # 4. Random inits
    for seed in range(3):
        inits.append((f'rand_{seed}', _random_init(N, seed=seed)))
    
    # Sort inits by their initial radius (best first) to prioritize promising starts
    def init_key(item):
        name, centers = item
        try:
            r = _compute_radius(centers)
            return -r
        except Exception:
            return 0.0
    
    inits.sort(key=init_key)
    
    # Optimize each init
    for i, (name, init) in enumerate(inits):
        if time.time() > deadline - 25.0:
            break
            
        time_remaining = deadline - time.time() - 10.0
        if time_remaining < 5.0:
            break
            
        try:
            opt_time = min(15.0, time_remaining)
            optimized = _optimize_packing(init, time_limit=opt_time)
            r_opt = _compute_radius(optimized)
            
            if r_opt > best_radius:
                best_radius = r_opt
                best_centers = optimized.copy()
                
        except Exception:
            continue
            
    # Final polish on the best result
    if best_centers is not None and time.time() < deadline - 5.0:
        try:
            polish_time = min(10.0, deadline - time.time() - 2.0)
            if polish_time > 2.0:
                polished = _final_polish(best_centers, time_limit=polish_time)
                r_polished = _compute_radius(polished)
                if r_polished > best_radius:
                    best_radius = r_polished
                    best_centers = polished.copy()
        except Exception:
            pass
            
    # Final validation and clipping
    if best_centers is None:
        best_centers = _grid_init(N)
        
    best_centers = np.clip(best_centers, -0.4999999, 0.4999999)
    
    return best_centers
