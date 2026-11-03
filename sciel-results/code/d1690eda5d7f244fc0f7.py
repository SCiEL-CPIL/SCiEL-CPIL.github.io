"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
_DEADLINE = 145.0
_T0 = time.time()


def _elapsed():
    return time.time() - _T0


def _compute_radii(centers):
    """
    Compute the maximum possible radius for each circle such that
    all circles are disjoint and inside the unit square.
    
    The radius of circle i is limited by:
    1. Distance to the nearest wall.
    2. Half the distance to the nearest other circle center.
    
    This ensures r_i + r_j <= d_ij for all pairs, so no overlap.
    """
    x = centers[:, 0]
    y = centers[:, 1]
    
    # Distance to walls
    dist_walls = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    
    radii = dist_walls.copy()
    
    # Distance to other circles
    # Compute pairwise distances
    # centers shape (N, 2)
    # diff[i, j] = centers[i] - centers[j]
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Set diagonal to infinity so it doesn't affect min
    np.fill_diagonal(dists, np.inf)
    
    # The constraint is r_i <= 0.5 * dist(i, j) for all j
    # So r_i is limited by 0.5 * min_j dist(i, j)
    min_neighbor_dists = np.min(dists, axis=1)
    radii_from_neighbors = 0.5 * min_neighbor_dists
    
    # Final radius is the minimum of wall distance and neighbor constraint
    radii = np.minimum(radii, radii_from_neighbors)
    
    return radii


def _score_from_centers(centers):
    """Negative sum of radii for minimization."""
    radii = _compute_radii(centers)
    return -np.sum(radii)


def _optimize_centers(centers, time_budget=5.0):
    """
    Optimize center coordinates to maximize sum of radii.
    Uses Nelder-Mead which handles non-smooth functions well.
    """
    start_time = time.time()
    
    # Flatten centers for scipy
    x0 = centers.flatten()
    
    def objective(x):
        c = x.reshape(N, 2)
        return _score_from_centers(c)
    
    # Nelder-Mead parameters
    # maxiter is not strictly time-based, so we check time in a wrapper or just set high and rely on deadline
    # However, scipy minimize doesn't have a built-in time limit.
    # We will run it and check if we are close to deadline.
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10})
        if time.time() - start_time > time_budget:
             # If it took too long, we might still have a good result in res.x
             pass
        return res.x.reshape(N, 2)
    except Exception:
        return centers


def construct():
    best_centers = None
    best_score = -np.inf
    
    # Initial grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side)
             for i in range(side) for j in range(side)]
    init_centers = np.array(cells[:N], dtype=float)
    
    rng = np.random.default_rng(42)
    
    # Strategy:
    # 1. Start with a jittered grid.
    # 2. Run local optimization.
    # 3. Do a few random restarts with larger perturbations if time permits.
    
    n_restarts = 5
    
    for restart in range(n_restarts):
        if _elapsed() > _DEADLINE - 10:
            break
            
        if restart == 0:
            # Grid start
            centers = init_centers.copy()
        else:
            # Jittered grid or random start
            if restart == 1:
                centers = init_centers + rng.uniform(-0.05, 0.05, (N, 2))
            elif restart == 2:
                centers = init_centers + rng.uniform(-0.1, 0.1, (N, 2))
            else:
                # Random start
                centers = rng.uniform(0.1, 0.9, (N, 2))
                
            # Ensure inside square
            centers = np.clip(centers, 0.0, 1.0)
            
            # Quick local refinement before the heavy optimizer
            # This helps Nelder-Mead start in a better basin
            centers = _optimize_centers(centers, time_budget=2.0)
        
        # Main optimization
        optimized_centers = _optimize_centers(centers, time_budget=15.0)
        
        # Calculate score
        radii = _compute_radii(optimized_centers)
        score = np.sum(radii)
        
        if score > best_score:
            best_score = score
            best_centers = optimized_centers.copy()
            
        # If we have a very good score, we might stop early
        if best_score > 1.25:
             break

    if best_centers is None:
        # Fallback to initial grid
        best_centers = init_centers
        radii = _compute_radii(best_centers)
        best_score = np.sum(radii)
        
    # Construct final array (x, y, radius)
    radii = _compute_radii(best_centers)
    result = np.column_stack((best_centers, radii))
    
    # Final validity check and repair if necessary (should be valid by construction)
    # But just in case of floating point errors, we can shrink radii slightly
    # Actually, the construction guarantees r_i + r_j <= d_ij.
    # Wall constraint is also enforced.
    
    return result
