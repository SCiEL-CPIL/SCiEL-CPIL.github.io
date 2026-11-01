import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi, QhullError

N = 50
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin

def _compute_radius(pts):
    """
    Compute the exact maximum radius for the given centers.
    Returns -1e6 if invalid (outside cube).
    """
    # Check bounds with a small tolerance
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return -1e6
    
    # Pairwise distances
    # (N, 1, 3) - (1, N, 3) -> (N, N, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(pts)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _soft_min_obj(pts, beta=50.0):
    """
    Smooth approximation of the negative minimum radius.
    We want to maximize min(dists, wall_dists).
    So we minimize -min(dists, wall_dists).
    
    Using log-sum-exp for soft-min:
    soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
    
    Here x consists of all pair distances/2 and wall distances.
    """
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # We only care about the lower triangle to save memory/time, 
    # but for N=50, full matrix is fine.
    # Mask out diagonal
    mask = ~np.eye(N, dtype=bool)
    pair_dists = dists[mask] / 2.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(pts)
    wall_dists_flat = wall_dists.ravel()
    
    # Combine all constraints
    # We want to maximize the minimum of these.
    # So we minimize the soft-min of these values.
    # soft_min = -1/beta * log(sum(exp(-beta * values)))
    
    # To avoid overflow/underflow, subtract max
    all_vals = np.concatenate([pair_dists, wall_dists_flat])
    
    # Clip values to prevent issues if they go negative (overlap)
    # If overlap, radius is negative. Soft min should reflect this.
    # exp(-beta * val) will be huge if val is negative.
    
    # For numerical stability:
    # log(sum(exp(-beta * x))) = -beta * max(x) + log(sum(exp(-beta * (x - max(x)))))
    # soft_min = max(x) - 1/beta * log(sum(exp(-beta * (x - max(x)))))
    
    max_val = np.max(all_vals)
    # If max_val is very negative, all are negative.
    # exp(-beta * (x - max_val)) = exp(beta * (max_val - x))
    # Since max_val is the max, (max_val - x) >= 0.
    # So exponents are non-negative. This is safe.
    
    exp_terms = np.exp(-beta * (all_vals - max_val))
    log_sum = np.log(np.sum(exp_terms))
    
    soft_min_val = max_val - (1.0 / beta) * log_sum
    
    # We want to MINIMIZE the negative radius.
    # Radius is approx soft_min_val.
    # So objective is -soft_min_val.
    return -soft_min_val

def _lloyd_step(pts, iterations=5):
    """
    Perform Lloyd-like relaxation using Voronoi centroids.
    This pushes points towards the center of their available space.
    """
    current_pts = pts.copy()
    for _ in range(iterations):
        try:
            vor = Voronoi(current_pts)
        except QhullError:
            # If Voronoi fails (e.g., degenerate points), break
            break
            
        # Get centroids of Voronoi regions
        # vor.regions gives indices into vor.vertices
        # vor.point_region gives the region index for each point
        new_pts = np.zeros_like(current_pts)
        
        for i in range(N):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if len(region) == 0:
                # Unbounded region (shouldn't happen if we add bounding box points, 
                # but we don't for simplicity. If it happens, keep original pos)
                new_pts[i] = current_pts[i]
                continue
            
            verts = vor.vertices[region]
            centroid = np.mean(verts, axis=0)
            
            # Clip to cube
            centroid = np.clip(centroid, -0.5, 0.5)
            new_pts[i] = centroid
            
        # Move towards centroid with a small step size to avoid instability
        # Full Lloyd move can oscillate. Damped Lloyd is safer.
        alpha = 0.5
        current_pts = (1 - alpha) * current_pts + alpha * new_pts
        current_pts = np.clip(current_pts, -0.5, 0.5)
        
    return current_pts

def _seed_fcc():
    """
    Generate an FCC lattice initialization.
    """
    fcc_pts = []
    # 3x3x3 cells of size 1/3 in [0,1]
    cell_size = 1.0 / 3.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                x0, y0, z0 = i * cell_size, j * cell_size, k * cell_size
                # FCC basis in unit cell
                fcc_pts.append([x0, y0, z0])
                fcc_pts.append([x0 + 0.5 * cell_size, y0 + 0.5 * cell_size, z0])
                fcc_pts.append([x0 + 0.5 * cell_size, y0, z0 + 0.5 * cell_size])
                fcc_pts.append([x0, y0 + 0.5 * cell_size, z0 + 0.5 * cell_size])
    
    fcc_pts = np.array(fcc_pts)
    # Shift to [-0.5, 0.5]
    fcc_pts -= 0.5
    
    # Filter points inside the cube
    mask = np.all(np.abs(fcc_pts) <= 0.5, axis=1)
    fcc_pts = fcc_pts[mask]
    
    # If we have more than N, select N points with largest minimum distance
    # Greedy selection
    if len(fcc_pts) > N:
        selected = []
        remaining = fcc_pts.copy()
        for _ in range(N):
            if not remaining.size:
                break
            # Pick the point that is furthest from already selected points
            if not selected:
                pt = remaining[0]
            else:
                sel_arr = np.array(selected)
                d = np.min(np.linalg.norm(remaining[:, np.newaxis, :] - sel_arr[np.newaxis, :, :], axis=2), axis=1)
                pt = remaining[np.argmax(d)]
            selected.append(pt)
            # Remove points too close to pt to ensure diversity
            d = np.linalg.norm(remaining - pt, axis=1)
            remaining = remaining[d > 0.1]
        fcc_pts = np.array(selected)
        
    # If we have less than N, pad with random points
    if len(fcc_pts) < N:
        n_pad = N - len(fcc_pts)
        pad = np.random.uniform(-0.45, 0.45, size=(n_pad, 3))
        fcc_pts = np.vstack([fcc_pts, pad])
        
    return fcc_pts[:N]

def _seed_grid():
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid_pts = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid_init = grid_pts[:N].copy()
    # Add small jitter to break symmetry
    grid_init += np.random.normal(0, 0.005, size=(N, 3))
    grid_init = np.clip(grid_init, -0.5, 0.5)
    return grid_init

def _seed_random():
    return np.random.uniform(-0.45, 0.45, size=(N, 3))

def construct():
    best_pts = None
    best_r = -np.inf
    
    # List of initializers
    initializers = [
        _seed_fcc,
        _seed_grid,
        _seed_random
    ]
    
    # Add a few random restarts if time permits
    num_extra_randoms = 2
    for _ in range(num_extra_randoms):
        initializers.append(_seed_random)
        
    # Optimization parameters
    # We use L-BFGS-B with bounds to handle the cube constraint naturally
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    for i, init_func in enumerate(initializers):
        if time.time() - START_TIME > TIME_LIMIT - 20.0:
            break
            
        init_pts = init_func()
        init_pts = np.clip(init_pts, -0.5, 0.5)
        
        # 1. Lloyd Relaxation
        # This is fast and helps equalize distances
        try:
            lloyd_pts = _lloyd_step(init_pts, iterations=10)
        except Exception:
            lloyd_pts = init_pts
            
        # Evaluate Lloyd result
        r_lloyd = _compute_radius(lloyd_pts)
        if r_lloyd > best_r:
            best_r = r_lloyd
            best_pts = lloyd_pts.copy()
            
        # 2. Gradient Descent on Soft-Min
        # Use the Lloyd result as the starting point for gradient descent
        # This allows the optimizer to fine-tune positions based on the smooth objective
        
        # We might want to try a few different betas or just stick to one.
        # A moderate beta like 20-50 works well.
        beta = 30.0
        
        try:
            res = minimize(
                _soft_min_obj, 
                lloyd_pts.ravel(), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            opt_pts = res.x.reshape(N, 3)
            opt_pts = np.clip(opt_pts, -0.5, 0.5)
            
            r_opt = _compute_radius(opt_pts)
            if r_opt > best_r:
                best_r = r_opt
                best_pts = opt_pts.copy()
                
        except Exception:
            pass
            
        # 3. Final Nelder-Mead Polish on the best found so far
        # NM doesn't use gradients, so it can handle the non-smoothness of the exact min
        # better than BFGS at the very end, provided we start from a good point.
        
    # If we have time, run a final polish on the best solution found
    if best_pts is not None and time.time() - START_TIME < TIME_LIMIT - 10.0:
        try:
            # Use exact radius for NM
            res = minimize(
                lambda x: -_compute_radius(x.reshape(N, 3)), 
                best_pts.ravel(), 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
            )
            nm_pts = res.x.reshape(N, 3)
            nm_pts = np.clip(nm_pts, -0.5, 0.5)
            
            r_nm = _compute_radius(nm_pts)
            if r_nm > best_r:
                best_r = r_nm
                best_pts = nm_pts.copy()
        except Exception:
            pass

    # Fallback if something went wrong
    if best_pts is None:
        best_pts = _seed_grid()

    return best_pts
