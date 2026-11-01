"""Circle packing in perimeter-4 rectangle: Grid-based cellular decomposition with variable cells."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
NX = 5
NY = 5
N_CELLS = NX * NY
ACTIVE_CELLS = list(range(N))
INACTIVE_CELLS = list(range(N, N_CELLS))

def _compute_cell_bounds(grid_x, grid_y):
    """
    For each cell, compute the range of valid center positions and max radius.
    Returns arrays: cx_min, cx_max, cy_min, cy_max, r_max
    """
    cx_min = np.zeros(N_CELLS)
    cx_max = np.zeros(N_CELLS)
    cy_min = np.zeros(N_CELLS)
    cy_max = np.zeros(N_CELLS)
    r_max = np.zeros(N_CELLS)
    
    idx = 0
    for r in range(NY):
        for c in range(NX):
            x_left = grid_x[c]
            x_right = grid_x[c + 1]
            y_bottom = grid_y[r]
            y_top = grid_y[r + 1]
            
            w_cell = x_right - x_left
            h_cell = y_top - y_bottom
            r_cell = min(w_cell, h_cell) / 2.0
            
            cx_min[idx] = x_left + r_cell
            cx_max[idx] = x_right - r_cell
            cy_min[idx] = y_bottom + r_cell
            cy_max[idx] = y_top - r_cell
            r_max[idx] = r_cell
            
            idx += 1
    
    return cx_min, cx_max, cy_min, cy_max, r_max

def _objective_full(z, w, h, grid_x, grid_y):
    """
    Full objective: optimize circle positions within cells.
    z layout: [dx_0, ..., dx_{N-1}, dy_0, ..., dy_{N-1}, r_0, ..., r_{N-1}]
    where dx, dy are offsets from cell center, r is radius.
    """
    n_vars = 3 * N
    dx = z[0:N]
    dy = z[N:2*N]
    r = z[2*N:3*N]
    
    # Cell centers
    cx0 = (grid_x[:-1] + grid_x[1:]) / 2.0
    cy0 = (grid_y[:-1] + grid_y[1:]) / 2.0
    cx0 = cx0.reshape(NY, NX).flatten()
    cy0 = cy0.reshape(NY, NX).flatten()
    
    active_idx = np.array(ACTIVE_CELLS)
    cx_cell = cx0[active_idx]
    cy_cell = cy0[active_idx]
    
    x = cx_cell + dx
    y = cy_cell + dy
    
    # Check boundaries
    if np.any(x < r) or np.any(x > w - r) or np.any(y < r) or np.any(y > h - r):
        return 1e6
    
    # Check non-negativity of radii
    if np.any(r < 1e-8):
        return 1e6
    
    # Overlap penalty
    dx_arr = x[:, None] - x[None, :]
    dy_arr = y[:, None] - y[None, :]
    d_sq = dx_arr * dx_arr + dy_arr * dy_arr
    r_sum = r[:, None] + r[None, :]
    
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
    r_sum_pairs = r_sum[iu]
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    
    penalty = 500.0 * np.sum(overlap * overlap)
    
    return -np.sum(r) + penalty

def _optimize_positions(grid_x, grid_y, w, h, time_budget):
    """Optimize circle positions and radii within fixed grid cells."""
    cx0 = (grid_x[:-1] + grid_x[1:]) / 2.0
    cy0 = (grid_y[:-1] + grid_y[1:]) / 2.0
    cx0 = cx0.reshape(NY, NX).flatten()
    cy0 = cy0.reshape(NY, NX).flatten()
    
    cx_min, cx_max, cy_min, cy_max, r_max = _compute_cell_bounds(grid_x, grid_y)
    
    active_idx = np.array(ACTIVE_CELLS)
    cx_cell = cx0[active_idx]
    cy_cell = cy0[active_idx]
    cx_min_a = cx_min[active_idx]
    cx_max_a = cx_max[active_idx]
    cy_min_a = cy_min[active_idx]
    cy_max_a = cy_max[active_idx]
    r_max_a = r_max[active_idx]
    
    # Initial guess: centered, max radius
    z0 = np.zeros(3 * N)
    z0[N:2*N] = 0.0  # dy
    z0[2*N:3*N] = r_max_a * 0.95
    
    bounds = []
    for i in range(N):
        bounds.append((cx_min_a[i] - cx_cell[i], cx_max_a[i] - cx_cell[i]))
    for i in range(N):
        bounds.append((cy_min_a[i] - cy_cell[i], cy_max_a[i] - cy_cell[i]))
    for i in range(N):
        bounds.append((1e-6, r_max_a[i]))
    
    def obj(z):
        return _objective_full(z, w, h, grid_x, grid_y)
    
    maxiter = int(min(200, time_budget * 30))
    
    try:
        res = minimize(obj, z0, method='SLSQP', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 50})
        z_opt = res.x if (res.success or res.fun < obj(z0)) else z0
    except Exception:
        z_opt = z0
    
    dx = z_opt[0:N]
    dy = z_opt[N:2*N]
    r = z_opt[2*N:3*N]
    
    x = cx_cell + dx
    y = cy_cell + dy
    
    return x, y, r

def _optimize_grid_lines(w, h, time_budget):
    """Optimize grid lines, then optimize positions within cells."""
    gx0 = np.linspace(w/NX, w - w/NX, NX - 1)
    gy0 = np.linspace(h/NY, h - h/NY, NY - 1)
    z0 = np.concatenate([gx0, gy0])
    
    bounds = []
    for _ in range(NX - 1):
        bounds.append((1e-4, w - 1e-4))
    for _ in range(NY - 1):
        bounds.append((1e-4, h - 1e-4))
    
    def obj_grid(z):
        gx = z[0:NX-1]
        gy = z[NX-1:NX-1+NY-1]
        
        if np.any(np.diff(gx) < 1e-5) or np.any(np.diff(gy) < 1e-5):
            return 1e6
        
        grid_x = np.concatenate(([0.0], gx, [w]))
        grid_y = np.concatenate(([0.0], gy, [h]))
        
        if np.any(np.diff(grid_x) < 1e-4) or np.any(np.diff(grid_y) < 1e-4):
            return 1e6
        
        # Quick estimate: sum of max radii in active cells
        cx0 = (grid_x[:-1] + grid_x[1:]) / 2.0
        cy0 = (grid_y[:-1] + grid_y[1:]) / 2.0
        cx0 = cx0.reshape(NY, NX).flatten()
        cy0 = cy0.reshape(NY, NX).flatten()
        
        w_cells = np.diff(grid_x)
        h_cells = np.diff(grid_y)
        w_cells_2d = np.tile(w_cells, NY)
        h_cells_2d = np.repeat(h_cells, NX)
        
        r_max_all = np.minimum(w_cells_2d, h_cells_2d) / 2.0
        
        active_idx = np.array(ACTIVE_CELLS)
        r_max_active = r_max_all[active_idx]
        
        # Penalty for inactive cells being large
        inactive_idx = np.array(INACTIVE_CELLS)
        r_max_inactive = r_max_all[inactive_idx]
        
        return -np.sum(r_max_active) + 5.0 * np.sum(r_max_inactive)
    
    maxiter = int(min(200, time_budget * 15))
    
    try:
        res = minimize(obj_grid, z0, method='SLSQP', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 50})
        z_opt = res.x if (res.success or res.fun < obj_grid(z0)) else z0
    except Exception:
        z_opt = z0
    
    gx_opt = z_opt[0:NX-1]
    gy_opt = z_opt[NX-1:NX-1+NY-1]
    
    grid_x = np.concatenate(([0.0], gx_opt, [w]))
    grid_y = np.concatenate(([0.0], gy_opt, [h]))
    
    # Now optimize positions
    time_left = time_budget - (time.time() - time.time())  # approximate
    x, y, r = _optimize_positions(grid_x, grid_y, w, h, time_budget * 0.5)
    
    return x, y, r, grid_x, grid_y

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
    
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.75, 1.33, 0.7, 1.5, 0.67, 0.85, 1.15, 1.2, 0.83, 1.4, 0.71]
    
    for w in aspect_ratios:
        if time.time() > deadline - 8:
            break
        h = 2.0 - w
        if h < 0.3:
            continue
        
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 3:
            break
        
        try:
            x, y, r, grid_x, grid_y = _optimize_grid_lines(w, h, remaining)
            
            valid, val = _check_validity(x, y, r, w, h)
            if valid and val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w)
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
