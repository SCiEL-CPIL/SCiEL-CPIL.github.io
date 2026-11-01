"""Circle packing in perimeter-4 rectangle: Grid-based cellular decomposition with variable cells."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _cell_params(grid_x, grid_y, active_mask):
    """
    Calculate centers and inradii for active cells.
    active_mask: boolean array of shape (NY, NX)
    Returns:
        centers: (N, 2) array of [x, y]
        radii: (N,) array of inradii
    """
    NY, NX = active_mask.shape
    centers = []
    radii = []
    
    for r in range(NY):
        for c in range(NX):
            if not active_mask[r, c]:
                continue
            x_left = grid_x[c]
            x_right = grid_x[c + 1]
            y_bottom = grid_y[r]
            y_top = grid_y[r + 1]
            
            cx = (x_left + x_right) / 2.0
            cy = (y_bottom + y_top) / 2.0
            
            w_cell = x_right - x_left
            h_cell = y_top - y_bottom
            r_cell = min(w_cell, h_cell) / 2.0
            
            centers.append([cx, cy])
            radii.append(r_cell)
            
    return np.array(centers), np.array(radii)

def _objective_grid(z, w, h, active_mask):
    """
    Objective function for grid optimization.
    z layout: [gx_0, ..., gx_{NX-2}, gy_0, ..., gy_{NY-2}]
    """
    NY, NX = active_mask.shape
    
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
        
    centers, radii = _cell_params(grid_x, grid_y, active_mask)
    
    # The count of centers should be N
    if len(centers) != N:
        return 1e6
        
    total_r = np.sum(radii)
    
    # Overlap penalty between active circles
    dx = centers[:, 0, None] - centers[None, :, 0]
    dy = centers[:, 1, None] - centers[None, :, 1]
    d_sq = dx * dx + dy * dy
    r_sum = radii[:, None] + radii[None, :]
    
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
    r_sum_pairs = r_sum[iu]
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    
    penalty = 1000.0 * np.sum(overlap * overlap)
    
    return -total_r + penalty

def _optimize_grid(w, h, active_mask, time_budget):
    """Optimize grid lines for given aspect ratio and mask."""
    NY, NX = active_mask.shape
    
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
        return _objective_grid(z, w, h, active_mask)
        
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
    
    centers, radii = _cell_params(grid_x, grid_y, active_mask)
    
    x = centers[:, 0]
    y = centers[:, 1]
    r = radii
    
    return x, y, r

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
    
    # Define topologies (NY, NX, mask)
    topologies = []
    
    # 1. 3x7 full grid
    mask_37 = np.ones((3, 7), dtype=bool)
    topologies.append((3, 7, mask_37))
    
    # 2. 7x3 full grid
    mask_73 = np.ones((7, 3), dtype=bool)
    topologies.append((7, 3, mask_73))
    
    # 3. 5x5 with 4 corners removed
    mask_55 = np.ones((5, 5), dtype=bool)
    mask_55[0,0] = False
    mask_55[0,4] = False
    mask_55[4,0] = False
    mask_55[4,4] = False
    topologies.append((5, 5, mask_55))
    
    # 4. 5x5 with 4 in a 2x2 block removed (top-left)
    mask_55_22 = np.ones((5, 5), dtype=bool)
    mask_55_22[0:2, 0:2] = False
    topologies.append((5, 5, mask_55_22))
    
    # 5. 4x6 with 3 corners removed
    mask_46 = np.ones((4, 6), dtype=bool)
    mask_46[0,0] = False
    mask_46[0,5] = False
    mask_46[3,0] = False
    topologies.append((4, 6, mask_46))
    
    # Aspect ratios to test for each topology
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.25, 0.75, 1.33, 0.7, 1.5, 0.67, 0.85, 1.15, 1.4, 0.6]
    
    for w in aspect_ratios:
        if time.time() > deadline - 5:
            break
        h = 2.0 - w
        if h < 0.3:
            continue
        
        remaining = deadline - (time.time() - start_time)
        if remaining < 2:
            break
        
        for NY, NX, mask in topologies:
            if time.time() > deadline - 2:
                break
            try:
                x, y, r = _optimize_grid(w, h, mask, remaining / len(topologies))
                
                valid, val = _check_validity(x, y, r, w, h)
                if valid and val > best_value:
                    best_value = val
                    best_result = (x.copy(), y.copy(), r.copy(), w)
            except Exception:
                pass
    
    if best_result is None:
        # Fallback to simple grid
        w, h = 1.0, 1.0
        x = []
        y = []
        r = []
        count = 0
        for i in range(5):
            for j in range(5):
                if count < 21:
                    x.append((i + 0.5) * 0.2)
                    y.append((j + 0.5) * 0.2)
                    r.append(0.09)
                    count += 1
        x = np.array(x)
        y = np.array(y)
        r = np.array(r)
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
