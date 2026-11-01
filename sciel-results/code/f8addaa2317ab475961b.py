"""Improved packing of 50 equal circles in a unit square via binary search and optimization."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

def _energy(centers, R):
    """
    Calculate the total overlap energy for circles of radius R.
    """
    centers = np.array(centers).reshape(-1, 2)
    
    # Pairwise overlaps
    if len(centers) < 2:
        pair_penalty = 0.0
    else:
        dists = squareform(pdist(centers, metric='euclidean'))
        np.fill_diagonal(dists, np.inf)
        viol = 2 * R - dists
        pair_penalty = np.sum(np.maximum(0, viol) ** 2)
    
    # Boundary penalties
    x = centers[:, 0]
    y = centers[:, 1]
    
    dist_left = x + 0.5
    dist_right = 0.5 - x
    dist_bottom = y + 0.5
    dist_top = 0.5 - y
    
    bound_penalty = np.sum(np.maximum(0, R - dist_left) ** 2) + \
                    np.sum(np.maximum(0, R - dist_right) ** 2) + \
                    np.sum(np.maximum(0, R - dist_bottom) ** 2) + \
                    np.sum(np.maximum(0, R - dist_top) ** 2)
    
    return pair_penalty + bound_penalty

def _find_packing(R, max_iter=300, n_restarts=3, tol=1e-6):
    """
    Attempt to pack N circles of radius R.
    Returns centers if successful, else None.
    """
    min_energy_found = np.inf
    best_centers = None
    
    for restart in range(n_restarts):
        np.random.seed(42 + restart)
        # Initial guess: Jittered Grid
        # 8x8 grid = 64 points. Pick 50.
        grid_x = np.linspace(-0.5 + R, 0.5 - R, 8)
        grid_y = np.linspace(-0.5 + R, 0.5 - R, 8)
        gx, gy = np.meshgrid(grid_x, grid_y)
        grid_points = np.column_stack((gx.ravel(), gy.ravel()))
        
        idx = np.random.choice(len(grid_points), N, replace=False)
        x0 = grid_points[idx]
        x0 += np.random.normal(0, 0.01, size=x0.shape)
        x0 = np.clip(x0, -0.5, 0.5)
        
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        res = minimize(_energy, x0, args=(R,), method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.fun < min_energy_found:
            min_energy_found = res.fun
            best_centers = res.x
            
        if min_energy_found < tol:
            return res.x
            
    if min_energy_found < tol:
        return best_centers
    return None

def construct():
    """Construct 50 circle centres in [-0.5, 0.5]^2 maximizing the common radius."""
    # Binary search for the maximum radius
    lo = 0.05
    hi = 0.08
    
    # Ensure hi is invalid
    # Quick check
    if _find_packing(hi, n_restarts=1, max_iter=100) is not None:
        while _find_packing(hi, n_restarts=1, max_iter=100) is not None:
            lo = hi
            hi += 0.001
            if hi > 0.1:
                break

    # Binary search
    for _ in range(20):
        mid = (lo + hi) / 2
        # Use fewer restarts and iterations for speed during search
        if _find_packing(mid, n_restarts=2, max_iter=200) is not None:
            lo = mid
        else:
            hi = mid
            
    # Final optimization with more restarts/iterations for the best radius
    centers = _find_packing(lo, n_restarts=5, max_iter=500)
    
    if centers is None:
        # Fallback
        grid_x = np.linspace(-0.4, 0.4, 7)
        grid_y = np.linspace(-0.4, 0.4, 7)
        gx, gy = np.meshgrid(grid_x, grid_y)
        centers = np.column_stack((gx.ravel(), gy.ravel()))[:N]
        
    return centers
# EVOLVE-BLOCK-END
