import numpy as np
import time
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 145.0  # 5s safety margin from 150s limit

def _get_W_H(W):
    H = 2.0 - W
    return W, H

def _calc_penalty_and_obj(x, smooth=True):
    """
    x: array of shape (64,)
    x[0] = W
    x[1:22] = cx
    x[22:43] = cy
    x[43:64] = r
    """
    W = x[0]
    H = 2.0 - W
    
    # If W or H is non-positive, return large penalty
    if W <= 0.1 or H <= 0.1:
        return 1e6, np.zeros(64)

    cx = x[1:22]
    cy = x[22:43]
    r = x[43:64]
    
    # Objective: -sum(r)
    obj = -np.sum(r)
    grad = np.zeros(64)
    grad[43:64] = -1.0
    
    # Penalty weight
    mu = 1000.0
    
    # Wall constraints
    d_left = cx
    d_bottom = cy
    d_right = W - cx
    d_top = H - cy
    
    d_min = np.minimum(np.minimum(d_left, d_bottom), np.minimum(d_right, d_top))
    viol_wall = r - d_min
    
    if smooth:
        eps = 1e-4
        # Use soft-max for smoothness
        viol_smooth = 0.5 * (viol_wall + np.sqrt(viol_wall**2 + eps**2))
        pen_wall = np.sum(viol_smooth**2)
        grad_viol = 0.5 * (1.0 + viol_wall / np.sqrt(viol_wall**2 + eps**2))
        d_pen_wall = 2.0 * viol_smooth * grad_viol
    else:
        viol_wall = np.maximum(viol_wall, 0.0)
        pen_wall = np.sum(viol_wall**2)
        d_pen_wall = 2.0 * viol_wall

    obj += mu * pen_wall
    
    # Gradient for wall penalty
    for i in range(N):
        if d_pen_wall[i] == 0:
            continue
        
        min_d = d_min[i]
        g_cx = 0.0
        g_cy = 0.0
        eps = 1e-9
        cnt = 0
        if d_left[i] <= min_d + eps:
            g_cx += 1.0
            cnt += 1
        if d_right[i] <= min_d + eps:
            g_cx += -1.0
            cnt += 1
        if d_bottom[i] <= min_d + eps:
            g_cy += 1.0
            cnt += 1
        if d_top[i] <= min_d + eps:
            g_cy += -1.0
            cnt += 1
        if cnt > 0:
            g_cx /= cnt
            g_cy /= cnt
            
        grad[1+i] -= mu * d_pen_wall[i] * g_cx
        grad[22+i] -= mu * d_pen_wall[i] * g_cy
        grad[43+i] += mu * d_pen_wall[i]

    # Overlap constraints
    for i in range(N):
        for j in range(i+1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.hypot(dx, dy)
            req = r[i] + r[j]
            viol = req - d
            
            if smooth:
                eps = 1e-4
                viol_smooth = 0.5 * (viol + np.sqrt(viol**2 + eps**2))
                pen = viol_smooth**2
                obj += mu * pen
                grad_viol = 0.5 * (1.0 + viol / np.sqrt(viol**2 + eps**2))
                d_pen = 2.0 * viol_smooth * grad_viol
            else:
                if viol <= 0:
                    continue
                pen = viol**2
                obj += mu * pen
                d_pen = 2.0 * viol
            
            if d < 1e-12:
                grad[1+i] += mu * d_pen * 1.0
                grad[1+j] += mu * d_pen * -1.0
                grad[43+i] += mu * d_pen * 1.0
                grad[43+j] += mu * d_pen * 1.0
            else:
                grad[1+i] += mu * d_pen * (-dx / d)
                grad[22+i] += mu * d_pen * (-dy / d)
                grad[1+j] += mu * d_pen * (dx / d)
                grad[22+j] += mu * d_pen * (dy / d)
                grad[43+i] += mu * d_pen * 1.0
                grad[43+j] += mu * d_pen * 1.0
                
    return obj, grad

def _evaluate(x):
    """
    Evaluate if x is valid. Return sum of radii or None.
    """
    W = x[0]
    H = 2.0 - W
    if W <= 0 or H <= 0:
        return None
        
    cx = x[1:22]
    cy = x[22:43]
    r = x[43:64]
    
    # Check walls
    for i in range(N):
        if r[i] > min(cx[i], cy[i], W-cx[i], H-cy[i]) + 1e-5:
            return None
            
    # Check overlaps
    for i in range(N):
        for j in range(i+1, N):
            d = np.hypot(cx[i]-cx[j], cy[i]-cy[j])
            if r[i] + r[j] > d + 1e-5:
                return None
                
    return float(np.sum(r))

def _hex_init(W, H, rows=5, cols=5):
    """
    Create a hexagonal grid initialization.
    """
    # Adjust rows/cols to fit 21 circles
    # 5x5 = 25, remove 4 corners or just take first 21
    cx_list = []
    cy_list = []
    
    # Simple grid with jitter
    x_coords = np.linspace(W/cols, W - W/cols, cols)
    y_coords = np.linspace(H/rows, H - H/rows, rows)
    
    for y in y_coords:
        for x in x_coords:
            cx_list.append(x)
            cy_list.append(y)
            
    # Shuffle to pick 21
    indices = np.random.permutation(len(cx_list))
    indices = indices[:N]
    
    cx = np.array([cx_list[i] for i in indices])
    cy = np.array([cy_list[i] for i in indices])
    
    # Initial radius: min distance to wall / 2, with some factor
    r = np.zeros(N)
    for i in range(N):
        d_wall = min(cx[i], cy[i], W-cx[i], H-cy[i])
        r[i] = d_wall * 0.8
        
    x = np.zeros(64)
    x[0] = W
    x[1:22] = cx
    x[22:43] = cy
    x[43:64] = r
    return x

def _project_valid(x):
    """
    Force x to be valid by shrinking radii and adjusting positions if necessary.
    This is a heuristic repair step.
    """
    W = x[0]
    H = 2.0 - W
    cx = x[1:22].copy()
    cy = x[22:43].copy()
    r = x[43:64].copy()
    
    # 1. Clamp positions to [0, W] and [0, H]
    cx = np.clip(cx, 0, W)
    cy = np.clip(cy, 0, H)
    
    # 2. Enforce wall constraints: r_i <= min(cx_i, cy_i, W-cx_i, H-cy_i)
    for i in range(N):
        max_r = min(cx[i], cy[i], W-cx[i], H-cy[i])
        if r[i] > max_r:
            r[i] = max_r
            
    # 3. Resolve overlaps iteratively (simple shrink)
    # We do a few passes of shrinking radii that overlap
    for _ in range(10):
        modified = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(cx[i]-cx[j], cy[i]-cy[j])
                req = r[i] + r[j]
                if req > d + 1e-6:
                    # Shrink both
                    excess = req - d
                    # Reduce r_i and r_j proportionally or just by half excess
                    shrink = excess / 2.0 + 1e-6
                    r[i] = max(0, r[i] - shrink)
                    r[j] = max(0, r[j] - shrink)
                    modified = True
        if not modified:
            break
            
    x[1:22] = cx
    x[22:43] = cy
    x[43:64] = r
    return x

def construct():
    best_val = 0.0
    best_x = None
    
    # Multiple starts with different W and initializations
    # W ranges from 0.8 to 1.2 (since H=2-W, this covers square-ish to rectangular)
    w_candidates = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    # Time budget per start
    total_time_budget = 130.0 # Leave 20s for safety
    num_starts = 5
    time_per_start = total_time_budget / num_starts
    
    for W in w_candidates:
        if time.time() > DEADLINE - 10:
            break
            
        H = 2.0 - W
        # Generate a few initializations for this W
        for _ in range(2): # 2 random inits per W
            if time.time() > DEADLINE - 10:
                break
            
            x0 = _hex_init(W, H)
            
            # Optimize
            try:
                res = minimize(
                    lambda x: _calc_penalty_and_obj(x, smooth=True)[0],
                    x0, 
                    jac=lambda x: _calc_penalty_and_obj(x, smooth=True)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6, 'disp': False}
                )
                
                if res.x is not None:
                    # Try to project to valid
                    x_proj = _project_valid(res.x)
                    val = _evaluate(x_proj)
                    
                    if val is not None and val > best_val:
                        best_val = val
                        best_x = x_proj
                        
            except Exception:
                pass

    # Fallback: if no valid solution found, use a very conservative grid
    if best_x is None:
        W = 1.0
        H = 1.0
        x = np.zeros(64)
        x[0] = W
        # 5x5 grid
        kx, ky = 5, 5
        cx = np.linspace(0, W, kx+2)[1:-1]
        cy = np.linspace(0, H, ky+2)[1:-1]
        idx = 0
        for yy in cy:
            for xx in cx:
                if idx >= N: break
                r = min(xx, yy, W-xx, H-yy) * 0.5 # Small radius to ensure no overlap
                x[1+idx] = xx
                x[22+idx] = yy
                x[43+idx] = r
                idx += 1
        best_x = _project_valid(x)
        best_val = _evaluate(best_x)
        
    if best_x is None:
        # Absolute last resort: single large circle? No, need 21.
        # Return 21 tiny circles in a grid
        W = 1.0
        H = 1.0
        x = np.zeros(64)
        x[0] = W
        for i in range(N):
            row = i // 5
            col = i % 5
            cx = (col + 0.5) * W / 5.0
            cy = (row + 0.5) * H / 5.0
            r = 0.01
            x[1+i] = cx
            x[22+i] = cy
            x[43+i] = r
        best_x = x
        best_val = 0.21

    # Final output formatting
    cx = best_x[1:22]
    cy = best_x[22:43]
    r = best_x[43:64]
    
    # Ensure strict validity one last time
    W = best_x[0]
    H = 2.0 - W
    cx = np.clip(cx, 0, W)
    cy = np.clip(cy, 0, H)
    r = np.maximum(r, 1e-6)
    
    for i in range(N):
        max_r = min(cx[i], cy[i], W-cx[i], H-cy[i])
        if r[i] > max_r:
            r[i] = max_r
            
    # Re-check overlaps and shrink if necessary (safety net)
    for _ in range(5):
        modified = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(cx[i]-cx[j], cy[i]-cy[j])
                req = r[i] + r[j]
                if req > d + 1e-6:
                    excess = req - d
                    shrink = excess / 2.0 + 1e-6
                    r[i] = max(1e-6, r[i] - shrink)
                    r[j] = max(1e-6, r[j] - shrink)
                    modified = True
        if not modified:
            break

    out = np.column_stack((cx, cy, r))
    return out
