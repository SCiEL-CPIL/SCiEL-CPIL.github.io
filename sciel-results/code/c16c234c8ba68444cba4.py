"""Rewritten program: local optimization of circle packing in a unit square."""
import numpy as np
import time

N = 200

# EVOLVE-BLOCK-START
def _score(centers):
    """Compute the minimum radius for a given set of centers."""
    # Boundary constraint
    boundary_min = np.min(0.5 - np.abs(centers))
    # Pairwise distances
    # Compute all pairwise differences
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Set diagonal to infinity
    np.fill_diagonal(dists, np.inf)
    pair_min = np.min(dists) / 2.0
    return min(boundary_min, pair_min)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    best_score = _score(centers)
    best_centers = centers.copy()
    
    # Local optimization using gradient-free methods
    # We'll use a simple iterative improvement: move each point slightly
    # in the direction that increases the minimum constraint
    
    # First, let's try a simple local search with small perturbations
    # and keep the best result
    
    # Better approach: use scipy's optimization if available
    try:
        from scipy.optimize import minimize
        import scipy
        has_scipy = True
    except ImportError:
        has_scipy = False
    
    if has_scipy:
        # Define objective: negative of the minimum radius (we want to maximize)
        def objective(x):
            c = x.reshape(N, 2)
            # Boundary constraint
            boundary_min = np.min(0.5 - np.abs(c))
            # Pairwise distances - use vectorized computation
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            pair_min = np.min(dists) / 2.0
            return -min(boundary_min, pair_min)
        
        # Use Nelder-Mead or BFGS with multiple restarts
        current = centers.copy()
        
        # Try several iterations of local optimization
        max_outer = 50
        for outer in range(max_outer):
            if time.time() > deadline:
                break
                
            # Flatten for scipy
            x0 = current.flatten()
            
            try:
                result = minimize(objective, x0, method='Nelder-Mead',
                                  options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
                new_centers = result.x.reshape(N, 2)
                
                # Validate
                if np.all(np.isfinite(new_centers)):
                    new_score = -objective(new_centers.flatten())
                    if new_score > best_score:
                        best_score = new_score
                        best_centers = new_centers.copy()
                        current = new_centers.copy()
            except Exception:
                pass
        
        # If we improved, do a final polish with tighter tolerance
        if time.time() < deadline - 10:
            try:
                result = minimize(objective, current.flatten(), method='Nelder-Mead',
                                  options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
                new_centers = result.x.reshape(N, 2)
                if np.all(np.isfinite(new_centers)):
                    new_score = -objective(new_centers.flatten())
                    if new_score > best_score:
                        best_score = new_score
                        best_centers = new_centers.copy()
            except Exception:
                pass
    
    # Fallback: simple local search if scipy not available or failed
    if time.time() < deadline - 5:
        step_size = 0.01
        for iteration in range(1000):
            if time.time() > deadline:
                break
            
            improved = False
            for i in range(N):
                old_pos = best_centers[i].copy()
                for dx in [-step_size, 0, step_size]:
                    for dy in [-step_size, 0, step_size]:
                        test = best_centers.copy()
                        test[i] = old_pos + np.array([dx, dy])
                        if np.all(np.abs(test) < 0.5):
                            s = _score(test)
                            if s > best_score + 1e-12:
                                best_centers = test.copy()
                                best_score = s
                                improved = True
                if improved:
                    break
            if not improved:
                break
    
    # Ensure valid output
    best_centers = np.clip(best_centers, -0.5 + 1e-10, 0.5 - 1e-10)
    return best_centers
# EVOLVE-BLOCK-END
