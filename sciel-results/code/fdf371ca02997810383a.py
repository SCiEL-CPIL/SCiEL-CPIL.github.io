"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize


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


def _objective_lbfgsb(x):
    """Objective function for L-BFGS-B: minimize negative radius."""
    c = x.reshape(-1, 2)
    # Clip to ensure validity during optimization steps
    c = np.clip(c, -0.5, 0.5)
    
    # Calculate radius
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    radius = min(min_pair_dist / 2.0, min_boundary)
    
    return -radius


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0
    
    # Try a range of spacings
    for d in np.linspace(0.06, 0.08, 10):
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
    return _initialize_grid()


def _initialize_grid():
    """Initialize with a grid pattern."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _optimize(c_init, max_time=20.0):
    """Run L-BFGS-B optimization starting from c_init."""
    start_time = time.time()
    
    x0 = c_init.flatten()
    
    # Bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (N * 2)
    
    # Use L-BFGS-B
    # maxiter is a safeguard, but we will check time in the callback
    result = minimize(
        _objective_lbfgsb, 
        x0, 
        method='L-BFGS-B', 
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-9, 'maxls': 50}
    )
    
    return result.x.reshape(-1, 2)


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    best_c = None
    best_r = 0.0
    
    # 1. Hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_r:
        best_r = r_hex
        best_c = c_hex.copy()
    
    # 2. Grid initialization
    c_grid = _initialize_grid()
    r_grid = _compute_radius(c_grid)
    if r_grid > best_r:
        best_r = r_grid
        best_c = c_grid.copy()
    
    # 3. Optimize the best initialization found so far
    if time.time() < deadline - 30:
        c_opt = _optimize(best_c, max_time=30)
        c_opt = np.clip(c_opt, -0.5, 0.5)
        r_opt = _compute_radius(c_opt)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
    
    # 4. Try a few random restarts if time permits
    np.random.seed(42)
    for i in range(5):
        if time.time() > deadline - 20:
            break
        
        # Random initialization, but maybe jittered grid to avoid bad local minima
        c_rand = c_grid + np.random.uniform(-0.01, 0.01, size=(N, 2))
        c_rand = np.clip(c_rand, -0.5, 0.5)
        
        r_rand = _compute_radius(c_rand)
        if r_rand > best_r:
            best_r = r_rand
            best_c = c_rand.copy()
        
        # Optimize this random start
        if time.time() < deadline - 10:
            c_opt = _optimize(c_rand, max_time=10)
            c_opt = np.clip(c_opt, -0.5, 0.5)
            r_opt = _compute_radius(c_opt)
            if r_opt > best_r:
                best_r = r_opt
                best_c = c_opt.copy()
        
        # Also try optimizing a purely random start
        if time.time() < deadline - 5:
            c_pure_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
            c_opt = _optimize(c_pure_rand, max_time=5)
            c_opt = np.clip(c_opt, -0.5, 0.5)
            r_opt = _compute_radius(c_opt)
            if r_opt > best_r:
                best_r = r_opt
                best_c = c_opt.copy()

    # Final safety check
    if best_c is None:
        best_c = c_grid
    
    best_c = np.clip(best_c, -0.5, 0.5)
    return best_c
# EVOLVE-BLOCK-END
