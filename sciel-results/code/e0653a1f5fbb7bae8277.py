import numpy as np
import time
import math
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def _penalty_objective(vars, W, lam_overlap, lam_bound):
    """
    vars: [x1, y1, r1, ..., xN, yN, rN]
    W: width
    Returns penalty value.
    """
    H = 2.0 - W
    x = vars[0:N]
    y = vars[N:2*N]
    r = vars[2*N:3*N]
    
    # Base objective: maximize sum of radii -> minimize -sum(r)
    val = -np.sum(r)
    
    # Overlap penalty
    overlap_pen = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d_sq = dx*dx + dy*dy
            min_d = r[i] + r[j]
            if d_sq < min_d*min_d:
                d = math.sqrt(d_sq)
                viol = min_d - d
                overlap_pen += viol * viol
    
    # Boundary penalty
    bound_pen = 0.0
    for i in range(N):
        # x violation
        if x[i] < r[i]:
            bound_pen += (r[i] - x[i])**2
        elif x[i] > W - r[i]:
            bound_pen += (x[i] - (W - r[i]))**2
            
        # y violation
        if y[i] < r[i]:
            bound_pen += (r[i] - y[i])**2
        elif y[i] > H - r[i]:
            bound_pen += (y[i] - (H - r[i]))**2
            
        # radius lower bound
        if r[i] < 1e-5:
            bound_pen += (1e-5 - r[i])**2

    return val + lam_overlap * overlap_pen + lam_bound * bound_pen

def _compute_value(x, y, r, W):
    """
    Returns sum of radii if valid, else 0.0
    """
    H = 2.0 - W
    if W <= 0 or H <= 0:
        return 0.0
        
    # Check bounds
    for i in range(N):
        if x[i] < r[i] - 1e-7 or x[i] > W - r[i] + 1e-7:
            return 0.0
        if y[i] < r[i] - 1e-7 or y[i] > H - r[i] + 1e-7:
            return 0.0
            
    # Check pairwise distances
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            min_dist = r[i] + r[j]
            if dist_sq < min_dist*min_dist - 1e-7:
                return 0.0
                
    return np.sum(r)

def _grid_init(W, H, n_cols, n_rows):
    """
    Initialize circles in a grid.
    """
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if idx >= N:
                break
            cx = (col + 0.5) * (W / n_cols)
            cy = (row + 0.5) * (H / n_rows)
            
            # Conservative radius
            dx_cell = W / n_cols
            dy_cell = H / n_rows
            r_max = 0.5 * min(dx_cell, dy_cell) * 0.95
            
            x[idx] = cx
            y[idx] = cy
            r[idx] = r_max
            idx += 1
        if idx >= N:
            break
            
    return x, y, r

def _run_optimization(x0, y0, r0, W0, deadline):
    """
    Run L-BFGS-B optimization with penalty.
    """
    H0 = 2.0 - W0
    vars0 = np.concatenate([x0, y0, r0])
    
    # Bounds for L-BFGS-B
    bounds = []
    for i in range(N):
        bounds.append((0.0, W0)) # x
    for i in range(N):
        bounds.append((0.0, H0)) # y
    for i in range(N):
        bounds.append((1e-5, 1.0)) # r
        
    lam_overlap = 100.0
    lam_bound = 100.0
    
    options = {
        'maxiter': 1000,
        'ftol': 1e-12,
        'gtol': 1e-12,
        'disp': False
    }
    
    start_time = time.time()
    
    try:
        res = minimize(
            lambda v: _penalty_objective(v, W0, lam_overlap, lam_bound),
            vars0,
            method='L-BFGS-B',
            bounds=bounds,
            options=options
        )
        
        vars_opt = res.x
        x_opt = vars_opt[0:N]
        y_opt = vars_opt[N:2*N]
        r_opt = vars_opt[2*N:3*N]
        
        # Validate
        val = _compute_value(x_opt, y_opt, r_opt, W0)
        
        if val > 0:
            return x_opt, y_opt, r_opt, W0, val
        else:
            return x0, y0, r0, W0, _compute_value(x0, y0, r0, W0)
            
    except Exception as e:
        return x0, y0, r0, W0, _compute_value(x0, y0, r0, W0)

def _polish(x, y, r, W, deadline):
    """
    Final high-precision polish pass with tighter tolerances.
    """
    H = 2.0 - W
    vars0 = np.concatenate([x, y, r])
    
    bounds = []
    for i in range(N):
        bounds.append((max(0.0, x[i] - 0.01), min(W, x[i] + 0.01)))
    for i in range(N):
        bounds.append((max(0.0, y[i] - 0.01), min(H, y[i] + 0.01)))
    for i in range(N):
        bounds.append((max(1e-5, r[i] - 0.01), min(1.0, r[i] + 0.01)))
        
    lam_overlap = 500.0
    lam_bound = 500.0
    
    options = {
        'maxiter': 2000,
        'ftol': 1e-15,
        'gtol': 1e-15,
        'disp': False
    }
    
    start_time = time.time()
    
    try:
        res = minimize(
            lambda v: _penalty_objective(v, W, lam_overlap, lam_bound),
            vars0,
            method='L-BFGS-B',
            bounds=bounds,
            options=options
        )
        
        vars_opt = res.x
        x_opt = vars_opt[0:N]
        y_opt = vars_opt[N:2*N]
        r_opt = vars_opt[2*N:3*N]
        
        val = _compute_value(x_opt, y_opt, r_opt, W)
        
        if val > 0:
            return x_opt, y_opt, r_opt, val
        else:
            return x, y, r, _compute_value(x, y, r, W)
            
    except Exception as e:
        return x, y, r, _compute_value(x, y, r, W)

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    best_val = 0.0
    best_x = np.zeros(N)
    best_y = np.zeros(N)
    best_r = np.zeros(N)
    best_W = 1.0
    
    # Try different aspect ratios
    aspect_ratios = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    
    # Grid configurations (cols, rows)
    grid_configs = [(3, 7), (4, 6), (5, 5), (6, 4), (7, 3), (3, 8), (8, 3), (4, 5), (5, 4)]
    
    # First pass: find a good aspect ratio
    for W0 in aspect_ratios:
        if time.time() > deadline - 20:
            break
        H0 = 2.0 - W0
        
        for n_cols, n_rows in grid_configs:
            if time.time() > deadline - 20:
                break
                
            x0, y0, r0 = _grid_init(W0, H0, n_cols, n_rows)
            
            # Add small noise
            x0 += np.random.uniform(-0.01, 0.01, N)
            y0 += np.random.uniform(-0.01, 0.01, N)
            r0 += np.random.uniform(-0.01, 0.01, N)
            
            # Clamp
            for i in range(N):
                r0[i] = max(1e-4, r0[i])
                x0[i] = np.clip(x0[i], r0[i], W0 - r0[i])
                y0[i] = np.clip(y0[i], r0[i], H0 - r0[i])
                
            x_opt, y_opt, r_opt, W_opt, val = _run_optimization(x0, y0, r0, W0, deadline)
            
            if val > best_val:
                best_val = val
                best_x = x_opt.copy()
                best_y = y_opt.copy()
                best_r = r_opt.copy()
                best_W = W_opt
    
    # Second pass: refine the best aspect ratio with more restarts
    if time.time() < deadline - 15 and best_val > 0:
        W_best = best_W
        H_best = 2.0 - W_best
        
        for _ in range(8):
            if time.time() > deadline - 15:
                break
                
            n_cols, n_rows = (5, 5) if W_best > 1.1 else (4, 6)
            x0, y0, r0 = _grid_init(W_best, H_best, n_cols, n_rows)
            
            x0 += np.random.uniform(-0.02, 0.02, N)
            y0 += np.random.uniform(-0.02, 0.02, N)
            r0 += np.random.uniform(-0.02, 0.02, N)
            
            for i in range(N):
                r0[i] = max(1e-4, r0[i])
                x0[i] = np.clip(x0[i], r0[i], W_best - r0[i])
                y0[i] = np.clip(y0[i], r0[i], H_best - r0[i])
                
            x_opt, y_opt, r_opt, W_opt, val = _run_optimization(x0, y0, r0, W_best, deadline)
            
            if val > best_val:
                best_val = val
                best_x = x_opt.copy()
                best_y = y_opt.copy()
                best_r = r_opt.copy()
                best_W = W_opt
    
    # Final polish pass
    if time.time() < deadline - 5 and best_val > 0:
        x_pol, y_pol, r_pol, val_pol = _polish(best_x, best_y, best_r, best_W, deadline)
        if val_pol > best_val:
            best_x = x_pol
            best_y = y_pol
            best_r = r_pol
            best_val = val_pol
    
    # Fallback
    if best_val == 0.0:
        W = 1.0
        H = 1.0
        side = 5
        for i in range(N):
            row = i // 5
            col = i % 5
            best_x[i] = (col + 0.5) / side
            best_y[i] = (row + 0.5) / side
            best_r[i] = 0.5 / side - 1e-4
        best_W = W
        
    return np.column_stack([best_x, best_y, best_r])

# EVOLVE-BLOCK-END
