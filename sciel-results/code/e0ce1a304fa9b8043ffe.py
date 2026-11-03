import numpy as np
from scipy.optimize import minimize
import time

N = 200

def _compute_radius(centres):
    """Compute the maximum common radius for the given centres."""
    # Distance between all pairs
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    # Distance to walls
    wall_dists = np.minimum(np.minimum(0.5 - centres[:, 0], 0.5 + centres[:, 0]),
                            np.minimum(0.5 - centres[:, 1], 0.5 + centres[:, 1]))
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _initial_hexagonal_layout(n):
    """Create an initial hexagonal-like layout for n circles."""
    best_centres = None
    best_radius = 0
    
    for rows in range(int(np.sqrt(n)) - 2, int(np.sqrt(n)) + 5):
        if rows <= 0:
            continue
        cols_per_row = np.ceil(n / rows).astype(int)
        
        centres = []
        for i in range(rows):
            for j in range(cols_per_row):
                if len(centres) >= n:
                    break
                x = (j + 0.5) / cols_per_row - 0.5
                y = (i + 0.5) / rows - 0.5
                if i % 2 == 1:
                    x += 0.5 / cols_per_row
                centres.append([x, y])
        
        if len(centres) == n:
            centres = np.array(centres)
            r = _compute_radius(centres)
            if r > best_radius:
                best_radius = r
                best_centres = centres
    
    if best_centres is None:
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centres = grid[:n]
    
    return best_centres

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Initial best solution
    best_centres = _initial_hexagonal_layout(N)
    best_radius = _compute_radius(best_centres)
    
    # Define objective and constraints for SLSQP
    # Variables: x = [x1, y1, ..., xN, yN, r]
    
    def objective(x):
        # Minimize -r
        return -x[-1]
        
    def constraints(x):
        centres = x[:-1].reshape(-1, 2)
        r = x[-1]
        
        # Pair constraints: d_ij >= 2r
        diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, 2*r) # Mask diagonal
        # We need d_ij - 2r >= 0
        # This returns a matrix of constraints. SLSQP expects a list of dicts or a single function returning array.
        # Flattening all pair constraints is huge (40000). This might be too slow for SLSQP.
        
        # Wall constraints: d_iw >= r
        wall_dists = np.minimum(np.minimum(0.5 - centres[:, 0], 0.5 + centres[:, 0]),
                                np.minimum(0.5 - centres[:, 1], 0.5 + centres[:, 1]))
        wall_cons = wall_dists - r
        
        # SLSQP can handle multiple constraints.
        # However, 40000 pair constraints is a lot.
        # Alternative: Maximize min radius without explicit constraints, using a penalty or just the min function.
        # Let's stick to the min function approach but with a better optimizer or strategy.
        
        return [] # Placeholder, we won't use SLSQP with all constraints due to size.

    # Revised Strategy: Use L-BFGS-B on the min-radius objective, but with "Nelder-Mead" as a fallback or hybrid.
    # Actually, the issue with L-BFGS-B on min-radius is the zero gradient.
    # Let's try to optimize the *sum* of distances to nearest neighbors first to spread them, then refine.
    
    def objective_spread(x):
        centres = x.reshape(-1, 2)
        diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        # Sum of min distances for each point
        min_dists = np.min(dist, axis=1)
        return -np.sum(min_dists**2) # Maximize sum of squared min distances
        
    def objective_radius(x):
        centres = x.reshape(-1, 2)
        r = _compute_radius(centres)
        return -r

    bounds = [(-0.5, 0.5)] * (2 * N)
    
    np.random.seed(42)
    
    # Try several random restarts
    for i in range(20):
        if time.time() > deadline - 20:
            break
            
        # Start from best known or random
        if i == 0:
            x0 = best_centres.flatten()
        else:
            # Perturb best known
            perturbation = np.random.randn(2 * N) * 0.05
            x0 = best_centres.flatten() + perturbation
            x0 = np.clip(x0, -0.5, 0.5)
        
        # Step 1: Spread out points
        try:
            res_spread = minimize(objective_spread, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100})
            if res_spread.success:
                x0 = res_spread.x
        except Exception:
            pass
            
        # Step 2: Maximize radius
        if time.time() > deadline - 10:
            continue
            
        try:
            res_rad = minimize(objective_radius, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12})
            if res_rad.success:
                new_centres = res_rad.x.reshape(-1, 2)
                new_radius = -res_rad.fun
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centres = new_centres.copy()
        except Exception:
            pass

    # Final check
    best_centres = np.clip(best_centres, -0.5, 0.5)
    return best_centres
