"""Circle packing in perimeter-4 rectangle: Grid-based cellular decomposition with local refinement."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _local_refine(x, y, r, w, h, time_budget=5.0):
    """
    Locally refine circle positions and radii within cell constraints.
    Each circle is constrained to stay within its original cell.
    We optimize all positions and radii simultaneously.
    """
    n_params = 3 * N
    x0 = np.concatenate([x, y, r])
    
    # Cell bounds for each circle (based on original grid)
    # We need to pass cell boundaries
    # For now, use conservative bounds: circle must stay within [r, w-r] x [r, h-r]
    # and radius must be positive and not exceed some max
    
    bounds = []
    for i in range(N):
        bounds.append((r[i] * 0.5, w - r[i] * 0.5))  # x_i
        bounds.append((r[i] * 0.5, h - r[i] * 0.5))  # y_i
        bounds.append((0.01, min(w, h) * 0.4))  # r_i
    
    def objective(z):
        xi = z[0:N]
        yi = z[N:2*N]
        ri = z[2*N:3*N]
        
        # Check boundary constraints
        if np.any(xi < ri - 1e-9) or np.any(xi > w - ri + 1e-9):
            return 1e6
        if np.any(yi < ri - 1e-9) or np.any(yi > h - ri + 1e-9):
            return 1e6
        
        # Overlap penalty
        dx = xi[:, None] - xi[None, :]
        dy = yi[:, None] - yi[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = ri[:, None] + ri[None, :]
        
        iu = np.triu_indices(N, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
        r_sum_pairs = r_sum[iu]
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        
        penalty = 500.0 * np.sum(overlap * overlap)
        
        return -np.sum(ri) + penalty
    
    start_t = time.time()
    maxiter = int(min(200, time_budget * 50))
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-10, 'maxls': 100})
        if time.time() - start_t < time_budget:
            if res.fun < objective(x0):
                x_new = res.x[0:N]
                y_new = res.x[N:2*N]
                r_new = res.x[2*N:3*N]
                return x_new, y_new, r_new
    except Exception:
        pass
    
    return x, y, r

def _grid_packing(w, h, nx, ny, time_budget=5.0):
    """
    Create a grid packing with nx x ny cells, each containing one circle.
    Optimizes grid line positions.
    """
    if nx * ny != N:
        return None
    
    # Grid lines: nx-1 vertical, ny-1 horizontal
    n_gx = nx - 1
    n_gy = ny - 1
    
    # Initial uniform grid
    gx0 = np.linspace(w / nx, w - w / nx, n_gx)
    gy0 = np.linspace(h / ny, h - h / ny, n_gy)
    z0 = np.concatenate([gx0, gy0])
    
    bounds = []
    for _ in range(n_gx):
        bounds.append((1e-3, w - 1e-3))
    for _ in range(n_gy):
        bounds.append((1e-3, h - 1e-3))
    
    def objective(z):
        gx = z[0:n_gx]
        gy = z[n_gx:n_gx + n_gy]
        
        if np.any(np.diff(gx) < 1e-4) or np.any(np.diff(gy) < 1e-4):
            return 1e6
        
        grid_x = np.concatenate(([0.0], gx, [w]))
        grid_y = np.concatenate(([0.0], gy, [h]))
        
        # Compute cell centers and inradii
        centers = np.zeros((N, 2))
        radii = np.zeros(N)
        
        idx = 0
        for row in range(ny):
            for col in range(nx):
                x_left = grid_x[col]
                x_right = grid_x[col + 1]
                y_bottom = grid_y[row]
                y_top = grid_y[row + 1]
                
                cx = (x_left + x_right) / 2.0
                cy = (y_bottom + y_top) / 2.0
                r_cell = min(x_right - x_left, y_top - y_bottom) / 2.0
                
                centers[idx] = [cx, cy]
                radii[idx] = r_cell
                idx += 1
        
        total_r = np.sum(radii)
        
        # Check overlaps (should be minimal in grid)
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
    
    maxiter = int(min(300, time_budget * 30))
    
    try:
        res = minimize(objective, z0, method='SLSQP', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 100})
        z_opt = res.x if (res.success or res.fun < objective(z0)) else z0
    except Exception:
        z_opt = z0
    
    gx_opt = z_opt[0:n_gx]
    gy_opt = z_opt[n_gx:n_gx + n_gy]
    
    grid_x = np.concatenate(([0.0], gx_opt, [w]))
    grid_y = np.concatenate(([0.0], gy_opt, [h]))
    
    # Compute final positions
    centers = np.zeros((N, 2))
    radii = np.zeros(N)
    
    idx = 0
    for row in range(ny):
        for col in range(nx):
            x_left = grid_x[col]
            x_right = grid_x[col + 1]
            y_bottom = grid_y[row]
            y_top = grid_y[row + 1]
            
            cx = (x_left + x_right) / 2.0
            cy = (y_bottom + y_top) / 2.0
            r_cell = min(x_right - x_left, y_top - y_bottom) / 2.0
            
            centers[idx] = [cx, cy]
            radii[idx] = r_cell
            idx += 1
    
    x = centers[:, 0]
    y = centers[:, 1]
    r = radii
    
    return x, y, r

def _check_validity(x, y, r, w, h, tol=1e-9):
    """Check if the packing is valid."""
    if w <= 0.05 or h <= 0.05:
        return False, 0.0
    if np.any(r <= 0):
        return False, 0.0
    if np.any(x < r - tol) or np.any(x > w - r + tol):
        return False, 0.0
    if np.any(y < r - tol) or np.any(y > h - r + tol):
        return False, 0.0
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
    r_sum_pairs = r_sum[iu]
    
    if np.any(d_pairs < r_sum_pairs - tol):
        return False, 0.0
    
    return True, float(np.sum(r))

def construct():
    start_time = time.time()
    deadline = start_time + 105.0
    
    best_value = 0.0
    best_result = None
    
    # Grid configurations: (nx, ny) where nx*ny == 21
    grid_configs = [
        (3, 7), (7, 3),
        (1, 21), (21, 1),
        (21, 1), (1, 21),
    ]
    
    # Aspect ratios to try
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.75, 1.33, 0.7, 1.5, 0.67, 0.85, 1.15, 0.95, 1.05]
    
    for w in aspect_ratios:
        if time.time() > deadline - 15:
            break
        h = 2.0 - w
        if h < 0.3:
            continue
        
        remaining = deadline - (time.time() - start_time)
        if remaining < 3:
            break
        
        for nx, ny in grid_configs:
            if time.time() > deadline - 10:
                break
            
            time_per = min(5.0, remaining / 3.0)
            
            try:
                result = _grid_packing(w, h, nx, ny, time_budget=time_per)
                if result is None:
                    continue
                
                x, y, r = result
                
                # Local refinement
                x, y, r = _local_refine(x, y, r, w, h, time_budget=min(3.0, remaining / 4.0))
                
                valid, val = _check_validity(x, y, r, w, h)
                if valid and val > best_value:
                    best_value = val
                    best_result = (x.copy(), y.copy(), r.copy(), w)
            except Exception:
                pass
    
    if best_result is None:
        # Fallback: simple 3x7 grid in unit square
        w, h = 1.0, 1.0
        nx, ny = 3, 7
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            col = i % nx
            row = i // nx
            x[i] = (col + 0.5) * w / nx
            y[i] = (row + 0.5) * h / ny
            r[i] = min(w / nx, h / ny) / 2.0 * 0.95
        
        valid, val = _check_validity(x, y, r, w, h)
        if valid:
            best_result = (x.copy(), y.copy(), r.copy(), w)
    
    if best_result is None:
        return np.array([[0.5, 0.5, 0.05]] * N)
    
    x, y, r, w = best_result
    h = 2.0 - w
    
    # Final slight shrink to ensure validity
    for _ in range(10):
        rt = r * (1 - 1e-7)
        if _check_validity(x, y, rt, w, h)[0]:
            r = rt
        else:
            break
            
    return np.column_stack((x, y, r))
