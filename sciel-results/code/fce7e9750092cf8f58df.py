"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit
    
    # Initial positions: simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    x0 = grid[:N].flatten()
    
    def compute_radius(x):
        """Compute the maximum radius for given positions."""
        positions = x.reshape(-1, 3)
        # Minimum distance to boundary
        boundary_dists = 0.5 - np.abs(positions)
        min_boundary = np.min(boundary_dists)
        
        # Minimum pairwise distance / 2
        from scipy.spatial.distance import pdist
        min_pair = np.min(pdist(positions)) / 2.0
        
        return min(min_boundary, min_pair)
    
    def negative_radius(x):
        """Objective function to minimize (negative of radius)."""
        return -compute_radius(x)
    
    # Initial radius
    r0 = compute_radius(x0)
    best_x = x0.copy()
    best_r = r0
    
    # Try multiple random restarts with local optimization
    rng = np.random.RandomState(42)
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() > deadline - 30.0:
            break
        
        # For first restart, use the grid; for others, perturb
        if restart == 0:
            x_current = x0.copy()
        else:
            x_current = x0 + rng.randn(N * 3) * 0.05
        
        # Local optimization using BFGS
        try:
            result = minimize(negative_radius, x_current, method='BFGS', 
                             options={'maxiter': 200, 'gtol': 1e-8})
            
            if result.success or result.fun < -best_r:
                r_new = -result.fun
                if r_new > best_r:
                    best_r = r_new
                    best_x = result.x.copy()
        except Exception:
            pass
    
    # Final polish: use COBYLA for more precise optimization
    if time.time() < deadline - 10.0:
        try:
            result = minimize(negative_radius, best_x, method='COBYLA',
                             options={'maxiter': 500, 'rhobeg': 1e-6})
            
            if result.fun < -best_r:
                best_r = -result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    return best_x.reshape(-1, 3)
# EVOLVE-BLOCK-END
