"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _soft_min(values, beta=50.0):
    """Compute a smooth approximation of the minimum using log-sum-exp."""
    # soft_min ≈ min - (1/beta) * log(sum(exp(-beta * (values - min))))
    # More numerically stable form:
    v_min = np.min(values)
    shifted = values - v_min
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return v_min - log_sum / beta


def _soft_objective(params, beta=50.0):
    """Minimize negative soft-min of distances (smooth surrogate)."""
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    
    # Pairwise distances / 2
    dists = pdist(c, metric='euclidean')
    pair_r = dists / 2.0
    
    # Boundary distances
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    # Combine all constraints into one vector for soft-min
    all_constraints = np.concatenate([pair_r, [min_boundary]])
    
    soft_r = _soft_min(all_constraints, beta)
    if soft_r <= 0:
        return 1.0
    return -soft_r


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern optimized for the square."""
    best_c = None
    best_r = 0
    
    for d in np.linspace(0.055, 0.085, 25):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is not None:
        return best_c
    
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _initialize_grid():
    """Initialize with a uniform grid pattern."""
    m = int(np.ceil(np.sqrt(N)))
    # Try different grid sizes
    for cols in range(m, m + 3):
        rows = int(np.ceil(N / cols))
        if rows * cols < N:
            continue
        
        x_ticks = (np.arange(cols) + 0.5) / cols - 0.5
        y_ticks = (np.arange(rows) + 0.5) / rows - 0.5
        
        grid = np.array([(x, y) for y in y_ticks for x in x_ticks])
        if len(grid) >= N:
            c = grid[:N].copy()
            r = _compute_radius(c)
            if r > 0:
                return c
    
    # Fallback
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _perturb(c, sigma=0.01):
    """Add random perturbation to centres."""
    c_pert = c.copy()
    noise = np.random.normal(0, sigma, size=c.shape)
    c_pert += noise
    c_pert = np.clip(c_pert, -0.5 + 1e-12, 0.5 - 1e-12)
    return c_pert


def _local_optimize(c0, max_time=25.0, max_iter=500):
    """Local optimization using L-BFGS-B with soft-min objective."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    bounds = [(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N)
    
    # First pass: use soft-min with moderate beta for smooth gradients
    try:
        result = minimize(
            lambda x: _soft_objective(x, beta=30.0),
            c0.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
        r_new = _compute_radius(c_new)
        if r_new > best_r:
            best_c = c_new
            best_r = r_new
    except Exception:
        pass
    
    # Second pass: increase beta for sharper minimum approximation
    if time.time() - start_time < max_time - 8:
        try:
            result2 = minimize(
                lambda x: _soft_objective(x, beta=80.0),
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': int(max_iter * 1.5), 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            c_new = np.clip(result2.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    
    # Third pass: very high beta for final polish
    if time.time() - start_time < max_time - 4:
        try:
            result3 = minimize(
                lambda x: _soft_objective(x, beta=200.0),
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': int(max_iter * 2), 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            c_new = np.clip(result3.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    
    return best_c, best_r


def _multi_restart_optimize(c_init, max_time=30.0, n_restarts=4):
    """Perform multiple restarts from perturbed initial positions."""
    start_time = time.time()
    best_c = c_init.copy()
    best_r = _compute_radius(c_init)
    
    # First optimize from the initial position with generous time
    remaining = max_time - (time.time() - start_time)
    c_opt, r_opt = _local_optimize(c_init, max_time=min(12.0, remaining - 5))
    if r_opt > best_r:
        best_c = c_opt
        best_r = r_opt
    
    # Try multiple restarts from perturbed versions
    for i in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > max_time - 10:
            break
        
        # Use decreasing perturbation sizes
        sigma = 0.015 / (i + 1)
        c_pert = _perturb(best_c, sigma=sigma)
        
        remaining = max_time - elapsed - 5
        c_opt, r_opt = _local_optimize(c_pert, max_time=min(8.0, remaining))
        if r_opt > best_r:
            best_c = c_opt
            best_r = r_opt
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_radius = 0.0
    best_c = None
    
    # Try hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
    
    # Try grid initialization
    c_grid = _initialize_grid()
    r_grid = _compute_radius(c_grid)
    if r_grid > best_radius:
        best_radius = r_grid
        best_c = c_grid.copy()
    
    # Choose the better initialization for optimization
    c_start = c_hex if r_hex >= r_grid else c_grid
    
    # Multi-restart optimization from the best initialization
    if time.time() < deadline - 15:
        c_opt, r_opt = _multi_restart_optimize(c_start, max_time=min(45.0, deadline - time.time() - 10))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Try a few structured perturbations if time permits
    np.random.seed(42)
    for trial in range(2):
        if time.time() > deadline - 25:
            break
        
        # Small random perturbation of the best solution
        c_pert = _perturb(best_c, sigma=0.005)
        r_pert = _compute_radius(c_pert)
        
        if r_pert > best_radius:
            best_radius = r_pert
            best_c = c_pert.copy()
        
        if time.time() < deadline - 15:
            c_opt, r_opt = _local_optimize(c_pert, max_time=min(12.0, deadline - time.time() - 5))
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    if best_c is None:
        best_c = _initialize_hexagonal()
    
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
