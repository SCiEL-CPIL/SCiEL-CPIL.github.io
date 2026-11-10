"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

def _surrogate_objective(flat_centres):
    """
    Objective function to minimize.
    We use a sum of inverse powers of distances to walls and between spheres.
    This creates a smooth landscape where the minimum corresponds to a packing
    that maximizes the minimum distance (as the terms with smallest distances dominate).
    """
    centres = flat_centres.reshape(-1, 3)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centres)
    # Avoid division by zero or negative values (though bounds should prevent this mostly)
    # Clip to a small epsilon to prevent inf/nan
    wall_dists = np.clip(wall_dists, 1e-6, None)
    wall_term = np.sum(wall_dists ** -10)
    
    # Pairwise distances
    # cdist is much faster than manual broadcasting for N=50
    dists = cdist(centres, centres)
    # Mask the diagonal
    np.fill_diagonal(dists, np.inf)
    # We only need the upper triangle to avoid double counting, but summing all is fine too if we divide by 2
    # Using all elements is simpler
    dists = np.clip(dists, 1e-6, None)
    pair_term = np.sum(dists ** -10)
    
    return wall_term + pair_term

def _compute_score(centres):
    """Compute the radius for a given set of centres."""
    n = len(centres)
    if n == 0:
        return 0.0
    # Wall constraint
    wall_dists = 0.5 - np.abs(centres)
    min_wall = np.min(wall_dists)
    # Pairwise distances
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_wall, min_pair)

def _optimize_local(centres, time_limit=5.0):
    """
    Optimize centres using scipy.optimize.minimize on the surrogate objective.
    """
    start_time = time.time()
    best_centres = centres.copy()
    best_obj = _surrogate_objective(centres.flatten())
    
    # We will try a few different methods or just one robust one.
    # Powell is derivative-free and robust for non-smooth-ish landscapes (though our surrogate is smooth).
    # L-BFGS-B is faster if gradients are good. Let's stick to Powell or Nelder-Mead for robustness without gradient implementation.
    # Actually, let's implement a simple gradient or just use Powell.
    
    # Flatten for scipy
    x0 = centres.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Use a tighter tolerance and maxiter to ensure we finish in time
    # Since N=50, 150 variables. Powell might be slow.
    # Let's try COBYLA or just a manual gradient ascent on the min-distance?
    # No, the directive suggests smooth surrogate.
    
    # To make it faster, we can limit the number of iterations.
    # We will run multiple short minimizations.
    
    for _ in range(3): # 3 restarts of local optimization with slight jitter
        if time.time() - start_time > time_limit:
            break
            
        # Jitter
        x0 = best_centres.flatten() + np.random.normal(0, 0.001, size=N*3)
        x0 = np.clip(x0, -0.499, 0.499)
        
        res = minimize(_surrogate_objective, x0, method='Powell', bounds=bounds, 
                       options={'maxiter': 200, 'xtol': 1e-8, 'ftol': 1e-8})
        
        if res.fun < best_obj:
            best_obj = res.fun
            best_centres = res.x.reshape(N, 3)
            
    return best_centres, _compute_score(best_centres)

def construct():
    """Construct 50 sphere centres in unit cube to maximize common radius."""
    start_time = time.time()
    deadline = 120.0  # seconds
    np.random.seed(42)
    
    best_centres = None
    best_score = 0.0
    
    # Initial configuration: Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    initial_grid = grid[:N].copy()
    
    # Try several random initializations and the grid
    initializations = [initial_grid]
    
    # Add random initializations
    for _ in range(10):
        random_centres = np.random.uniform(-0.49, 0.49, size=(N, 3))
        initializations.append(random_centres)
        
    # Also try a BCC-like initialization if possible, but random is often good enough with good optimizer
    # Let's stick to grid + randoms
    
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() - start_time > deadline - 10.0:
            break
            
        if i < len(initializations):
            centres = initializations[i].copy()
        else:
            # More random restarts
            centres = np.random.uniform(-0.49, 0.49, size=(N, 3))
            
        # Time budget per restart
        remaining = deadline - 10.0 - (time.time() - start_time)
        time_per_restart = max(1.0, remaining / (num_restarts - i))
        
        optimized, score = _optimize_local(centres, time_limit=time_per_restart)
        
        if score > best_score:
            best_score = score
            best_centres = optimized.copy()
            
    # Final polish: If we have time, do a very tight local search on the best found
    if best_centres is not None and time.time() - start_time < deadline - 2.0:
        # Use a higher precision objective or just run Powell again with more iterations
        x0 = best_centres.flatten()
        bounds = [(-0.5, 0.5)] * (N * 3)
        res = minimize(_surrogate_objective, x0, method='Powell', bounds=bounds, 
                       options={'maxiter': 500, 'xtol': 1e-10, 'ftol': 1e-10})
        if res.fun < _surrogate_objective(best_centres.flatten()):
            best_centres = res.x.reshape(N, 3)
            best_score = _compute_score(best_centres)
            
    # Ensure bounds
    best_centres = np.clip(best_centres, -0.499999, 0.499999)
    
    return best_centres
# EVOLVE-BLOCK-END
