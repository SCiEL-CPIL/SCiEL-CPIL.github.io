"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_radius(centres):
    """Compute the maximum common radius for given centres."""
    n = len(centres)
    # Minimum distance between any pair of centres
    min_pair_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centres[i] - centres[j])
            if d < min_pair_dist:
                min_pair_dist = d
    # Minimum distance from any centre to cube boundary
    min_boundary_dist = np.inf
    for i in range(n):
        for k in range(3):
            d = 0.5 - abs(centres[i, k])
            if d < min_boundary_dist:
                min_boundary_dist = d
    return min(min_pair_dist / 2.0, min_boundary_dist)


def _compute_radius_fast(centres):
    """Vectorized radius computation for speed."""
    n = len(centres)
    # Pairwise distances using broadcasting (memory intensive for large N but N=100 is ok)
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to boundaries
    boundary_dists = 0.5 - np.abs(centres)
    min_boundary_dist = np.min(boundary_dists)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)


def _objective(centres_flat):
    """Objective function to minimize (negative radius)."""
    centres = centres_flat.reshape(-1, 3)
    # Check validity
    if not np.all(np.isfinite(centres)):
        return 1e10
    if np.any(np.abs(centres) > 0.5):
        return 1e10
    r = _compute_radius_fast(centres)
    return -r


def _local_optimize(centres, time_limit=10.0):
    """Run local optimization to improve the packing."""
    start_time = time.time()
    n = len(centres)
    centres = centres.copy()
    
    # Use Nelder-Mead or Powell for optimization
    # Flatten centres for scipy
    x0 = centres.flatten()
    
    best_centres = centres.copy()
    best_radius = _compute_radius_fast(best_centres)
    
    # Try multiple optimization methods
    methods = ['Nelder-Mead', 'Powell']
    
    for method in methods:
        if time.time() - start_time > time_limit:
            break
            
        try:
            result = minimize(
                _objective, x0, method=method,
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            if result.fun < 0:  # Valid solution
                new_centres = result.x.reshape(-1, 3)
                new_radius = _compute_radius_fast(new_centres)
                
                if new_radius > best_radius:
                    best_centres = new_centres
                    best_radius = new_radius
        except Exception:
            pass
    
    return best_centres


def _initial_grid():
    """Generate initial cubic grid."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N]


def construct():
    """Construct the optimal packing of 100 spheres in a unit cube."""
    start_time = time.time()
    deadline = start_time + 120  # Must finish within 120s
    
    # Start with cubic grid
    centres = _initial_grid()
    best_centres = centres.copy()
    best_radius = _compute_radius_fast(best_centres)
    
    # Local optimization
    try:
        optimized = _local_optimize(centres, time_limit=60.0)
        opt_radius = _compute_radius_fast(optimized)
        
        if opt_radius > best_radius:
            best_centres = optimized
            best_radius = opt_radius
    except Exception:
        pass
    
    # If we have time, try with different initializations
    if time.time() < deadline - 30:
        # Try random initialization
        np.random.seed(42)
        random_centres = np.random.uniform(-0.45, 0.45, (N, 3))
        
        try:
            optimized_rand = _local_optimize(random_centres, time_limit=30.0)
            rand_radius = _compute_radius_fast(optimized_rand)
            
            if rand_radius > best_radius:
                best_centres = optimized_rand
                best_radius = rand_radius
        except Exception:
            pass
    
    # Ensure validity: clip to cube boundaries
    best_centres = np.clip(best_centres, -0.5, 0.5)
    
    return best_centres
# EVOLVE-BLOCK-END
