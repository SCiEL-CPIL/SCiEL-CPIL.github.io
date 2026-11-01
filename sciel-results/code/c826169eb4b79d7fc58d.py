"""Circle packing in perimeter-4 rectangle: Grid-based cellular decomposition with free positioning."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
NX = 5
NY = 5
N_CELLS = NX * NY
ACTIVE_CELLS = list(range(N))
INACTIVE_CELLS = list(range(N, N_CELLS))


def _build_initial_grid(w, h):
    """Build initial uniform grid."""
    gx = np.linspace(w / NX, w - w / NX, NX - 1)
    gy = np.linspace(h / NY, h - h / NY, NY - 1)
    return gx, gy


def _cell_bounds(grid_x, grid_y):
    """
    For each active cell, return the bounding box [x_min, x_max, y_min, y_max].
    Only active cells (first N) are returned.
    """
    boxes = np.zeros((N, 4))
    idx = 0
    for r in range(NY):
        for c in range(NX):
            if idx < N:
                x_left = grid_x[c]
                x_right = grid_x[c + 1]
                y_bottom = grid_y[r]
                y_top = grid_y[r + 1]
                boxes[idx] = [x_left, x_right, y_bottom, y_top]
            idx += 1
    return boxes


def _objective(z, w, h):
    """
    Objective function.
    z layout:
      [0..NX-2]   : internal x grid lines (NX-1 values)
      [NX-1..NX-1+NY-2] : internal y grid lines (NY-1 values)
      [NX-1+NY-1 .. ]   : for each active circle: dx, dy, r (3*N values)
        where dx, dy are offsets from cell center, r is radius
    """
    n_gx = NX - 1
    n_gy = NY - 1
    n_grid = n_gx + n_gy
    n_circ = 3 * N
    
    gx = z[0:n_gx]
    gy = z[n_gx:n_gx + n_gy]
    circ_params = z[n_grid:n_grid + n_circ]
    
    # Check grid validity
    if np.any(np.diff(gx) < 1e-5) or np.any(np.diff(gy) < 1e-5):
        return 1e8
    
    grid_x = np.concatenate(([0.0], gx, [w]))
    grid_y = np.concatenate(([0.0], gy, [h]))
    
    if np.any(np.diff(grid_x) < 1e-4) or np.any(np.diff(grid_y) < 1e-4):
        return 1e8
    
    # Compute cell centers and max possible radii (inradius)
    centers = np.zeros((N, 2))
    max_radii = np.zeros(N)
    
    idx = 0
    for r in range(NY):
        for c in range(NX):
            if idx < N:
                x_left = grid_x[c]
                x_right = grid_x[c + 1]
                y_bottom = grid_y[r]
                y_top = grid_y[r + 1]
                
                cx = (x_left + x_right) / 2.0
                cy = (y_bottom + y_top) / 2.0
                w_cell = x_right - x_left
                h_cell = y_top - y_bottom
                r_max = min(w_cell, h_cell) / 2.0
                
                centers[idx] = [cx, cy]
                max_radii[idx] = r_max
            idx += 1
    
    # Extract circle parameters
    dx = circ_params[0:N]
    dy = circ_params[N:2*N]
    radii = circ_params[2*N:3*N]
    
    # Actual positions
    x = centers[:, 0] + dx
    y = centers[:, 1] + dy
    
    # Penalty for circles going outside their cell
    cell_pen = 0.0
    for i in range(N):
        r = radii[i]
        if r < 1e-6:
            cell_pen += 100.0
            continue
        # Check if circle fits in its cell
        # The circle center is at (x[i], y[i]) with radius r
        # It must be within the cell bounds
        # Cell bounds for circle i:
        idx_cell = i
        c = idx_cell % NX
        r_cell = idx_cell // NX
        x_left = grid_x[c]
        x_right = grid_x[c + 1]
        y_bottom = grid_y[r_cell]
        y_top = grid_y[r_cell + 1]
        
        # Circle must fit: x[i] - r >= x_left, x[i] + r <= x_right, etc.
        if x[i] - r < x_left - 1e-6:
            cell_pen += 100.0 * (x_left - (x[i] - r)) ** 2
        if x[i] + r > x_right + 1e-6:
            cell_pen += 100.0 * ((x[i] + r) - x_right) ** 2
        if y[i] - r < y_bottom - 1e-6:
            cell_pen += 100.0 * (y_bottom - (y[i] - r)) ** 2
        if y[i] + r > y_top + 1e-6:
            cell_pen += 100.0 * ((y[i] + r) - y_top) ** 2
    
    # Overlap penalty between circles
    dx_arr = x[:, None] - x[None, :]
    dy_arr = y[:, None] - y[None, :]
    d_sq = dx_arr * dx_arr + dy_arr * dy_arr
    r_sum = radii[:, None] + radii[None, :]
    
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
    r_sum_pairs = r_sum[iu]
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    
    overlap_pen = 1000.0 * np.sum(overlap * overlap)
    
    total_r = np.sum(radii)
    
    return -total_r + cell_pen + overlap_pen


def _optimize_grid(w, h, time_budget):
    """Optimize grid lines and circle parameters for given aspect ratio."""
    gx0, gy0 = _build_initial_grid(w, h)
    
    # Initial circle parameters: centered in cells, radius = 0.9 * inradius
    dx0 = np.zeros(N)
    dy0 = np.zeros(N)
    
    # Compute initial max radii
    grid_x0 = np.concatenate(([0.0], gx0, [w]))
    grid_y0 = np.concatenate(([0.0], gy0, [h]))
    
    r0 = np.zeros(N)
    idx = 0
    for r in range(NY):
        for c in range(NX):
            if idx < N:
                w_cell = grid_x0[c + 1] - grid_x0[c]
                h_cell = grid_y0[r + 1] - grid_y0[r]
                r0[idx] = 0.9 * min(w_cell, h_cell) / 2.0
            idx += 1
    
    z0 = np.concatenate([gx0, gy0, dx0, dy0, r0])
    
    # Bounds
    bounds = []
    # Grid x lines: must be sorted and within (0, w)
    for _ in range(NX - 1):
        bounds.append((1e-4, w - 1e-4))
    for _ in range(NY - 1):
        bounds.append((1e-4, h - 1e-4))
    # Circle offsets: bounded by cell half-dimensions
    # We'll use loose bounds and rely on penalties
    for _ in range(3 * N):
        bounds.append((-w, w))
    
    # Tighten radius bounds
    for i in range(2*N, 3*N):
        bounds[i] = (1e-6, min(w, h) / 2.0)
    
    def obj(z):
        return _objective(z, w, h)
    
    maxiter = int(min(500, time_budget * 30))
    
    try:
        res = minimize(obj, z0, method='SLSQP', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 100})
        z_opt = res.x if (res.success or res.fun < obj(z0)) else z0
    except Exception:
        z_opt = z0
    
    # Extract results
    n_gx = NX - 1
    n_gy = NY - 1
    n_grid = n_gx + n_gy
    
    gx_opt = z_opt[0:n_gx]
    gy_opt = z_opt[n_gx:n_gx + n_gy]
    circ_params = z_opt[n_grid:n_grid + 3*N]
    
    dx = circ_params[0:N]
    dy = circ_params[N:2*N]
    radii = circ_params[2*N:3*N]
    
    grid_x = np.concatenate(([0.0], gx_opt, [w]))
    grid_y = np.concatenate(([0.0], gy_opt, [h]))
    
    # Compute centers
    centers = np.zeros((N, 2))
    idx = 0
    for r in range(NY):
        for c in range(NX):
            if idx < N:
                cx = (grid_x[c] + grid_x[c + 1]) / 2.0
                cy = (grid_y[r] + grid_y[r + 1]) / 2.0
                centers[idx] = [cx, cy]
            idx += 1
    
    x = centers[:, 0] + dx
    y = centers[:, 1] + dy
    
    return x, y, radii, grid_x, grid_y


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


def _local_refine(x, y, r, w, h, time_budget):
    """Local refinement using direct circle optimization."""
    z0 = np.concatenate([x, y, r])
    
    bounds = []
    for i in range(N):
        bounds.append((r[i], w - r[i]))  # x
    for i in range(N):
        bounds.append((r[i], h - r[i]))  # y
    for i in range(N):
        bounds.append((1e-6, min(w, h) / 2.0))  # r
    
    def obj(z):
        x_c = z[0:N]
        y_c = z[N:2*N]
        r_c = z[2*N:3*N]
        
        # Overlap penalty
        dx = x_c[:, None] - x_c[None, :]
        dy = y_c[:, None] - y_c[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r_c[:, None] + r_c[None, :]
        
        iu = np.triu_indices(N, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
        r_sum_pairs = r_sum[iu]
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        
        pen = 1000.0 * np.sum(overlap * overlap)
        
        return -np.sum(r_c) + pen
    
    maxiter = int(min(300, time_budget * 20))
    
    try:
        res = minimize(obj, z0, method='SLSQP', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 50})
        if res.success or res.fun < obj(z0):
            z_opt = res.x
            x_new = z_opt[0:N]
            y_new = z_opt[N:2*N]
            r_new = z_opt[2*N:3*N]
            return x_new, y_new, r_new
    except Exception:
        pass
    
    return x, y, r


def construct():
    start_time = time.time()
    deadline = start_time + 105.0
    
    best_value = 0.0
    best_result = None
    
    # Try multiple aspect ratios
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.75, 1.33, 0.7, 1.5, 0.67]
    
    for w in aspect_ratios:
        if time.time() > deadline - 15:
            break
        h = 2.0 - w
        if h < 0.3:
            continue
        
        elapsed = time.time() - start_time
        remaining = deadline - (start_time + elapsed)
        if remaining < 3:
            break
        
        try:
            # Phase 1: Grid optimization
            x, y, r, grid_x, grid_y = _optimize_grid(w, h, remaining * 0.6)
            
            valid, val = _check_validity(x, y, r, w, h)
            if valid:
                # Phase 2: Local refinement
                if time.time() < deadline - 5:
                    x2, y2, r2 = _local_refine(x, y, r, w, h, remaining * 0.3)
                    valid2, val2 = _check_validity(x2, y2, r2, w, h)
                    if valid2 and val2 > val:
                        x, y, r, val = x2, y2, r2, val2
                
                if val > best_value:
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
