"""Circle packing in perimeter-4 rectangle: Grid-based cellular decomposition."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
NX = 5
NY = 5
N_CELLS = NX * NY
ACTIVE_CELLS = list(range(N))
INACTIVE_CELLS = list(range(N, N_CELLS))

def _compute_cell_radii(grid_x, grid_y):
    """Compute inradii for all cells. Circle centered in each cell."""
    radii = np.zeros(N_CELLS)
    idx = 0
    for r in range(NY):
        for c in range(NX):
            w_cell = grid_x[c + 1] - grid_x[c]
            h_cell = grid_y[r + 1] - grid_y[r]
            radii[idx] = min(w_cell, h_cell) / 2.0
            idx += 1
    return radii

def _compute_centers(grid_x, grid_y):
    """Compute centers for all cells."""
    centers = np.zeros((N_CELLS, 2))
    idx = 0
    for r in range(NY):
        for c in range(NX):
            cx = (grid_x[c] + grid_x[c + 1]) / 2.0
            cy = (grid_y[r] + grid_y[r + 1]) / 2.0
            centers[idx] = [cx, cy]
            idx += 1
    return centers

def _objective(z, w, h):
    """
    Objective: maximize sum of active cell inradii.
    z = [gx_0..gx_{NX-2}, gy_0..gy_{NY-2}]
    Since circles are centered in disjoint cells, no overlap is possible.
    """
    n_gx = NX - 1
    n_gy = NY - 1
    gx = z[0:n_gx]
    gy = z[n_gx:n_gx + n_gy]
    
    # Check sorting and minimum spacing
    if np.any(np.diff(gx) < 1e-6) or np.any(np.diff(gy) < 1e-6):
        return 1e6
    
    grid_x = np.concatenate(([0.0], gx, [w]))
    grid_y = np.concatenate(([0.0], gy, [h]))
    
    if np.any(np.diff(grid_x) < 1e-5) or np.any(np.diff(grid_y) < 1e-5):
        return 1e6
    
    radii = _compute_cell_radii(grid_x, grid_y)
    active_radii = radii[ACTIVE_CELLS]
    inactive_radii = radii[INACTIVE_CELLS]
    
    total_r = np.sum(active_radii)
    # Penalize inactive cells to encourage them to be small
    inactive_penalty = 5.0 * np.sum(inactive_radii)
    
    return -total_r + inactive_penalty

def _optimize_for_aspect(w, h, time_budget):
    """Optimize grid lines for given rectangle dimensions."""
    gx0 = np.linspace(w / NX, w - w / NX, NX - 1)
    gy0 = np.linspace(h / NY, h - h / NY, NY - 1)
    z0 = np.concatenate([gx0, gy0])
    
    bounds = [(1e-5, w - 1e-5)] * (NX - 1) + [(1e-5, h - 1e-5)] * (NY - 1)
    
    def obj(z):
        return _objective(z, w, h)
    
    maxiter = int(min(500, time_budget * 30))
    
    best_z = z0.copy()
    best_val = obj(z0)
    
    # Try multiple starting points
    n_starts = min(8, int(time_budget * 2) + 1)
    rng = np.random.default_rng(42)
    
    for start_i in range(n_starts):
        if time.time() > time_budget:
            break
        
        if start_i == 0:
            z_start = z0.copy()
        else:
            # Perturbed initial guess
            z_start = z0 + rng.normal(0, 0.05 * max(w, h), size=len(z0))
            z_start = np.clip(z_start, 1e-4, max(w, h) - 1e-4)
        
        try:
            res = minimize(obj, z_start, method='SLSQP', bounds=bounds,
                          options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})
            if res.fun < best_val:
                best_val = res.fun
                best_z = res.x.copy()
        except Exception:
            pass
    
    gx_opt = best_z[0:NX - 1]
    gy_opt = best_z[NX - 1:NX - 1 + NY - 1]
    
    grid_x = np.concatenate(([0.0], gx_opt, [w]))
    grid_y = np.concatenate(([0.0], gy_opt, [h]))
    
    centers = _compute_centers(grid_x, grid_y)
    radii = _compute_cell_radii(grid_x, grid_y)
    
    active_idx = np.array(ACTIVE_CELLS)
    x = centers[active_idx, 0]
    y = centers[active_idx, 1]
    r = radii[active_idx]
    
    return x, y, r

def _validate(x, y, r, w, h, tol=1e-9):
    """Validate the packing."""
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
    
    # Try various aspect ratios
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.67, 0.5, 0.9, 1.1]
    
    for w in aspect_ratios:
        if time.time() > deadline - 15:
            break
        h = 2.0 - w
        if h < 0.25:
            continue
        
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 3:
            break
        
        try:
            x, y, r = _optimize_for_aspect(w, h, remaining)
            valid, val = _validate(x, y, r, w, h)
            if valid and val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w)
        except Exception:
            pass
    
    if best_result is None:
        # Fallback: uniform 5x5 grid, use first 21 cells
        w, h = 1.0, 1.0
        x = np.array([(i % 5) * 0.2 + 0.1 for i in range(N)])
        y = np.array([(i // 5) * 0.2 + 0.1 for i in range(N)])
        r = np.full(N, 0.095)
        valid, val = _validate(x, y, r, w, h)
        if valid:
            best_result = (x.copy(), y.copy(), r.copy(), w)
    
    if best_result is None:
        return np.array([[0.1, 0.1, 0.05]] * N)
    
    x, y, r, w = best_result
    h = 2.0 - w
    
    # Slight shrink for safety
    for _ in range(3):
        rt = r * (1 - 1e-7)
        if _validate(x, y, rt, w, h)[0]:
            r = rt
        else:
            break
    
    return np.column_stack((x, y, r))
