"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist
import time

N = 50

# EVOLVE-BLOCK-START

def _compute_radius(pts):
    """Compute the actual packing radius for a set of points."""
    # Check bounds
    if not np.all(np.isfinite(pts)):
        return 0.0
    if pts.shape != (N, 2):
        return 0.0
    if np.any(pts[:, 0] < -0.5) or np.any(pts[:, 0] > 0.5) or np.any(pts[:, 1] < -0.5) or np.any(pts[:, 1] > 0.5):
        return 0.0
    # Compute pairwise distances
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    # Compute distance to walls
    wall_dist = np.minimum(
        np.minimum(pts[:, 0] + 0.5, 0.5 - pts[:, 0]),
        np.minimum(pts[:, 1] + 0.5, 0.5 - pts[:, 1])
    )
    min_wall = np.min(wall_dist)
    return min(min_pair / 2.0, min_wall)


def _smooth_objective(x, beta=50.0):
    """Smooth surrogate objective using log-sum-exp approximation of min distance.
    We maximize the minimum distance, so minimize negative smooth min."""
    pts = x.reshape(N, 2)
    # Pairwise distances
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    # Wall distances
    wall = np.minimum(
        np.minimum(pts[:, 0] + 0.5, 0.5 - pts[:, 0]),
        np.minimum(pts[:, 1] + 0.5, 0.5 - pts[:, 1])
    )
    # Combine all distances (pairwise/2 and wall)
    pair_half = d / 2.0
    all_dists = np.concatenate([pair_half.ravel(), wall.ravel()])
    all_dists = all_dists[all_dists > 1e-12]
    if len(all_dists) == 0:
        return 0.0
    # Log-sum-exp approximation of min
    m = np.min(all_dists)
    log_sum = np.log(np.sum(np.exp(-beta * (all_dists - m))))
    smooth_min = m - np.log(len(all_dists)) / beta - log_sum / beta
    # We want to maximize smooth_min, so minimize negative
    return -smooth_min


def _initial_grid():
    """Create initial grid arrangement."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _initial_random(seed=42):
    """Random initial points inside the square."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 2))


def _local_search(pts, max_iter=500, tol=1e-10):
    """Local optimization using L-BFGS-B with smooth objective."""
    x0 = pts.flatten()
    bounds = [(-0.4999, 0.4999)] * (2 * N)
    
    best_x = x0.copy()
    best_val = _smooth_objective(best_x)
    
    for beta in [20.0, 50.0, 100.0, 200.0]:
        res = minimize(_smooth_objective, best_x, method='L-BFGS-B', 
                      bounds=bounds, args=(beta,),
                      options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x.copy()
    
    return best_x.reshape(N, 2)


def _differential_evolution_search(bounds, max_iter=100, popsize=15, seed=42):
    """Use differential evolution for global search."""
    res = differential_evolution(_smooth_objective, bounds, 
                                  maxiter=max_iter, popsize=popsize,
                                  seed=seed, tol=1e-8, 
                                  polish=False, updating='immediate')
    return res.x.reshape(N, 2)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with grid
    best_pts = _initial_grid()
    best_r = _compute_radius(best_pts)
    
    # Local search from grid
    if time.time() < deadline:
        refined = _local_search(best_pts)
        r = _compute_radius(refined)
        if r > best_r:
            best_pts = refined
            best_r = r
    
    # Try multiple random starts with local search
    rng = np.random.RandomState(0)
    for i in range(10):
        if time.time() >= deadline - 30.0:
            break
        init = rng.uniform(-0.45, 0.45, size=(N, 2))
        refined = _local_search(init, max_iter=300)
        r = _compute_radius(refined)
        if r > best_r:
            best_r = r
            best_pts = refined
    
    # If we have time, try differential evolution
    if time.time() < deadline - 60.0:
        bounds = [(-0.4999, 0.4999)] * (2 * N)
        de_pts = _differential_evolution_search(bounds, max_iter=50, popsize=10, seed=123)
        r = _compute_radius(de_pts)
        if r > best_r:
            best_r = r
            best_pts = de_pts
        # Local refine the DE result
        if time.time() < deadline - 30.0:
            refined = _local_search(de_pts, max_iter=300)
            r = _compute_radius(refined)
            if r > best_r:
                best_r = r
                best_pts = refined
    
    # Final local refinement on best
    if time.time() < deadline - 10.0:
        refined = _local_search(best_pts, max_iter=500)
        r = _compute_radius(refined)
        if r > best_r:
            best_r = r
            best_pts = refined
    
    return best_pts

# EVOLVE-BLOCK-END
