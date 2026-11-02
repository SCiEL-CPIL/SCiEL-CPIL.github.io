"""Circle packing in perimeter-4 rectangle: Grid-based cellular decomposition with variable cells."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _cell_params(grid_x, grid_y, NX, NY):
    """
    Calculate centers and inradii for all cells.
    Returns:
        centers: (N_CELLS, 2) array of [x, y]
        radii: (N_CELLS,) array of inradii
    """
    N_CELLS = NX * NY
    centers = np.zeros((N_CELLS, 2))
    radii = np.zeros(N_CELLS)
    
    idx = 0
    for r in range(NY):
        for c in range(NX):
            x_left = grid_x[c]
            x_right = grid_x[c + 1]
            y_bottom = grid_y[r]
            y_top = grid_y[r + 1]
            
            cx = (x_left + x_right) / 2.0
            cy = (y_bottom + y_top) / 2.0
            
            # Inradius is half the min dimension
            w_cell = x_right - x_left
            h_cell = y_top - y_bottom
            r_cell = min(w_cell, h_cell) / 2.0
            
            centers[idx] = [cx, cy]
            radii[idx] = r_cell
            idx += 1
            
    return centers, radii

def _objective_grid(z, w, h, NX, NY, K_inactive):
    """
    Objective function for grid optimization.
    z layout: [gx_0, ..., gx_{NX-2}, gy_0, ..., gy_{NY-2}]
    """
    gx = z[0:NX-1]
    gy = z[NX-1:NX-1+NY-1]
    
    # Ensure grid lines are within bounds and sorted
    if np.any(np.diff(gx) < 1e-6) or np.any(np.diff(gy) < 1e-6):
        return 1e6
    
    grid_x = np.concatenate(([0.0], gx, [w]))
    grid_y = np.concatenate(([0.0], gy, [h]))
    
    # Check if any cell is too small (to avoid numerical issues)
    if np.any(np.diff(grid_x) < 1e-4) or np.any(np.diff(grid_y) < 1e-4):
        return 1e6
        
    centers, radii = _cell_params(grid_x, grid_y, NX, NY)
    
    # If K_inactive == 0, we use all cells.
    # If K_inactive > 0, we want to maximize the sum of the top N radii.
    # Approximation: Sum(all radii) - Lambda * Sum(smallest K_inactive radii)
    # This encourages the smallest cells to shrink to 0.
    
    if K_inactive == 0:
        total_r = np.sum(radii)
        inactive_penalty = 0.0
    else:
        # Sort radii to find the smallest K_inactive
        sorted_radii = np.sort(radii)
        smallest_sum = np.sum(sorted_radii[:K_inactive])
        total_r = np.sum(radii)
        # Penalty weight: should be high enough to force shrinkage but not dominate
        # If we just maximize total_r, uniform is best.
        # We want N large, K small.
        # Using a penalty on the smallest K works well.
        inactive_penalty = 10.0 * smallest_sum
    
    # Overlap penalty is technically not needed if we trust the grid geometry,
    # but it's a good safety net in case of numerical drift or if my geometric proof is flawed.
    # However, for grid-centered circles, overlaps are impossible if cells are disjoint.
    # So we can skip the O(N^2) overlap check to save time, or keep it with low weight.
    # Let's keep a small penalty just in case.
    active_centers = centers
    active_radii = radii
    n_cells = NX * NY
    dx = active_centers[:, 0, None] - active_centers[None, :, 0]
    dy = active_centers[:, 1, None] - active_centers[None, :, 1]
    d_sq = dx * dx + dy * dy
    r_sum = active_radii[:, None] + active_radii[None, :]
    
    iu = np.triu_indices(n_cells, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
    r_sum_pairs = r_sum[iu]
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    penalty = 100.0 * np.sum(overlap * overlap)
    
    return -total_r + penalty + inactive_penalty

def _optimize_grid(w, h, NX, NY, time_budget):
    """Optimize grid lines for given aspect ratio and grid dimensions."""
    N_CELLS = NX * NY
    K_inactive = N_CELLS - N
    
    # Initial guess: uniform grid
    gx0 = np.linspace(w/NX, w - w/NX, NX - 1)
    gy0 = np.linspace(h/NY, h - h/NY, NY - 1)
    
    z0 = np.concatenate([gx0, gy0])
    
    bounds = []
    for _ in range(NX - 1):
        bounds.append((1e-4, w - 1e-4))
    for _ in range(NY - 1):
        bounds.append((1e-4, h - 1e-4))
        
    def obj(z):
        return _objective_grid(z, w, h, NX, NY, K_inactive)
        
    # Use SLSQP for small dimension problem
    maxiter = int(min(300, time_budget * 20))
    
    try:
        res = minimize(obj, z0, method='SLSQP', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 100})
        z_opt = res.x if (res.success or res.fun < obj(z0)) else z0
    except Exception:
        z_opt = z0
        
    gx_opt = z_opt[0:NX-1]
    gy_opt = z_opt[NX-1:NX-1+NY-1]
    
    grid_x = np.concatenate(([0.0], gx_opt, [w]))
    grid_y = np.concatenate(([0.0], gy_opt, [h]))
    
    centers, radii = _cell_params(grid_x, grid_y, NX, NY)
    
    # Select the best N circles
    # If N_CELLS == N, take all.
    # Else, take the N largest radii.
    if N_CELLS == N:
        indices = np.arange(N)
    else:
        # Get indices of the N largest radii
        indices = np.argsort(radii)[::-1][:N]
        # Sort indices to keep a consistent order? Not necessary, but good for debugging.
        # We just need the centers and radii corresponding to these indices.
        
    x = centers[indices, 0]
    y = centers[indices, 1]
    r = radii[indices]
    
    return x, y, r, w

def _check_validity(x, y, r, w, h, tol=1e-9):
    """Check if the packing is valid."""
    if w <= 0.05 or h <= 0.05:
        return False, 0.0
    for i in range(N):
        if r[i] <= 0:
            return False, 0.0
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False, 0.0
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False, 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            r_sum = r[i] + r[j]
            if d2 < r_sum * r_sum - tol:
                return False, 0.0
    return True, float(np.sum(r))

def construct():
    start_time = time.time()
    deadline = start_time + 105.0
    
    best_value = 0.0
    best_result = None
    
    # List of grid shapes (NX, NY)
    grid_shapes = [(3, 7), (7, 3), (4, 6), (6, 4), (5, 5)]
    
    # Aspect ratios to try
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.75, 1.33, 0.7, 1.5, 0.67, 0.85, 1.15, 0.6, 1.67]
    
    for NX, NY in grid_shapes:
        if time.time() > deadline - 10:
            break
            
        for w in aspect_ratios:
            if time.time() > deadline - 2:
                break
            h = 2.0 - w
            if h < 0.3:
                continue
            
            remaining = deadline - (time.time() - start_time)
            if remaining < 2:
                break
            
            try:
                x, y, r, w_opt = _optimize_grid(w, h, NX, NY, remaining)
                
                valid, val = _check_validity(x, y, r, w_opt, h)
                if valid and val > best_value:
                    best_value = val
                    best_result = (x.copy(), y.copy(), r.copy(), w_opt)
            except Exception:
                pass
    
    if best_result is None:
        # Fallback to simple grid
        w, h = 1.0, 1.0
        x = np.array([(i % 5) * 0.2 + 0.1 for i in range(N)])
        y = np.array([(i // 5) * 0.2 + 0.1 for i in range(N)])
        r = np.full(N, 0.09)
        valid, val = _check_validity(x, y, r, w, h)
        if valid:
            best_result = (x.copy(), y.copy(), r.copy(), w)
    
    if best_result is None:
        return np.array([[0.1, 0.1, 0.05]] * N)
    
    x, y, r, w = best_result
    h = 2.0 - w
    
    # Final slight shrink to ensure validity
    for _ in range(5):
        rt = r * (1 - 1e-8)
        if _check_validity(x, y, rt, w, h)[0]:
            r = rt
        else:
            break
            
    return np.column_stack((x, y, r))
