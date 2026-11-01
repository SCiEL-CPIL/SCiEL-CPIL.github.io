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


def _get_distance_components(c):
    """Get all distance constraints (pairwise and boundary) as a flat array."""
    n = c.shape[0]
    pair_dists = pdist(c, metric='euclidean') / 2.0
    bound_dists = 0.5 - np.abs(c)
    all_dists = np.concatenate([pair_dists, bound_dists.ravel()])
    return all_dists


def _soft_min_objective(params, beta):
    """Minimize negative soft-min of distances."""
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    dists = _get_distance_components(c)
    dists = np.maximum(dists, 1e-12)
    
    if beta > 10000:
        return -np.min(dists)
    
    d_min = np.min(dists)
    shifted = dists - d_min
    terms = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(terms))
    soft_min_val = d_min - log_sum / beta
    return -soft_min_val


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern optimized for the square."""
    best_c = None
    best_r = 0
    
    for d in np.linspace(0.058, 0.082, 25):
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


def _local_optimize(c0, max_time=25.0):
    """Local optimization using L-BFGS-B with annealed soft-min."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    betas = [10.0, 50.0, 200.0, 1000.0]
    current_c = c0.copy()
    
    for beta in betas:
        if time.time() > start_time + max_time - 5:
            break
        
        try:
            result = minimize(
                _soft_min_objective,
                current_c.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
                current_c = c_new.copy()
        except Exception:
            pass
    
    if time.time() < start_time + max_time - 2:
        try:
            def _objective_raw(params):
                c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
                r = _compute_radius(c)
                return -r
            
            result = minimize(
                _objective_raw,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new.copy()
                best_r = r_new
        except Exception:
            pass
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_radius = 0.0
    best_c = None
    
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
    
    if time.time() < deadline - 10:
        c_opt, r_opt = _local_optimize(c_hex, max_time=min(100.0, deadline - time.time() - 5))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    if time.time() < deadline - 20:
        np.random.seed(42)
        c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
        r_rand = _compute_radius(c_rand)
        
        if r_rand > best_radius:
            best_radius = r_rand
            best_c = c_rand.copy()
        
        if time.time() < deadline - 15:
            c_opt, r_opt = _local_optimize(c_rand, max_time=min(15.0, deadline - time.time() - 5))
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    if best_c is None:
        best_c = _initialize_hexagonal()
    
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
