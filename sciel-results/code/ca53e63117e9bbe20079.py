import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 21
START = time.time()
DEADLINE = START + 140.0  # 10s safety margin

def _get_W_H(W):
    H = 2.0 - W
    return W, H

def _calc_penalty_and_obj(x, smooth=False):
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
    # d_i = min(cx_i, cy_i, W-cx_i, H-cy_i)
    # viol_i = max(r_i - d_i, 0)
    
    d_left = cx
    d_bottom = cy
    d_right = W - cx
    d_top = H - cy
    
    d_min = np.minimum(np.minimum(d_left, d_bottom), np.minimum(d_right, d_top))
    
    viol_wall = r - d_min
    
    if smooth:
        eps = 1e-3
        # smooth max
        viol_smooth = 0.5 * (viol_wall + np.sqrt(viol_wall**2 + eps**2))
        pen_wall = np.sum(viol_smooth**2)
        # Gradient of smooth max: 0.5 * (1 + viol / sqrt(viol^2 + eps^2))
        grad_viol = 0.5 * (1.0 + viol_wall / np.sqrt(viol_wall**2 + eps**2))
        d_pen_wall = 2.0 * viol_smooth * grad_viol
    else:
        viol_wall = np.maximum(viol_wall, 0.0)
        pen_wall = np.sum(viol_wall**2)
        d_pen_wall = 2.0 * viol_wall
    
    obj += mu * pen_wall
    
    # Gradient for wall penalty
    # d/dx of (r - d_min)^2
    # Need subgradient of d_min
    for i in range(N):
        if d_pen_wall[i] == 0:
            continue
        
        # Determine active wall
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
            
        # d viol / d cx = -g_cx
        # d viol / d r = 1
        # grad[1+i] += mu * d_pen_wall[i] * (-g_cx)
        # grad[22+i] += mu * d_pen_wall[i] * (-g_cy)
        # grad[43+i] += mu * d_pen_wall[i] * 1.0
        
        grad[1+i] -= mu * d_pen_wall[i] * g_cx
        grad[22+i] -= mu * d_pen_wall[i] * g_cy
        grad[43+i] += mu * d_pen_wall[i]

    # Overlap constraints
    # For each pair (i, j), viol = max(r_i + r_j - d_ij, 0)
    cx = x[1:22]
    cy = x[22:43]
    r = x[43:64]
    
    for i in range(N):
        for j in range(i+1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.hypot(dx, dy)
            req = r[i] + r[j]
            viol = req - d
            
            if smooth:
                eps = 1e-3
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
                # Degenerate, push apart
                grad[1+i] += mu * d_pen * 1.0
                grad[1+j] += mu * d_pen * -1.0
                grad[43+i] += mu * d_pen * 1.0
                grad[43+j] += mu * d_pen * 1.0
            else:
                # d viol / d cx_i = -dx/d
                # d viol / d cx_j = dx/d
                # d viol / d r_i = 1
                # d viol / d r_j = 1
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
        if r[i] > min(cx[i], cy[i], W-cx[i], H-cy[i]) + 1e-6:
            return None
            
    # Check overlaps
    for i in range(N):
        for j in range(i+1, N):
            d = np.hypot(cx[i]-cx[j], cy[i]-cy[j])
            if r[i] + r[j] > d + 1e-6:
                return None
                
    return float(np.sum(r))

def _grid_init(W, H):
    """
    Create a simple grid initialization.
    """
    kx, ky = 5, 5 # 25 spots, pick 21
    cx = np.linspace(0, W, kx+2)[1:-1]
    cy = np.linspace(0, H, ky+2)[1:-1]
    pts = []
    for yy in cy:
        for xx in cx:
            r = min(xx, yy, W-xx, H-yy) * 0.9
            pts.append([xx, yy, r])
    # Shuffle and take 21
    np.random.shuffle(pts)
    pts = pts[:N]
    
    x = np.zeros(64)
    x[0] = W
    for i in range(N):
        x[1+i] = pts[i][0]
        x[22+i] = pts[i][1]
        x[43+i] = pts[i][2]
    return x

def construct():
    best_val = 0.0
    best_x = None
    
    # Bounds for DE
    # W in [0.6, 1.4]
    # cx, cy in [0, 2]
    # r in [0.01, 0.5]
    bounds = [(0.6, 1.4)]
    for _ in range(N):
        bounds.append((0.0, 2.0)) # cx
    for _ in range(N):
        bounds.append((0.0, 2.0)) # cy
    for _ in range(N):
        bounds.append((0.01, 0.5)) # r
        
    # Run DE
    # Time budget: 60s
    de_start = time.time()
    de_deadline = de_start + 60.0
    
    try:
        # Use a callback to stop if time runs out? DE doesn't support callback for time easily in all versions.
        # We can just set maxiter reasonably low and popsize moderate.
        # Or run in a loop? No, DE is a single call.
        # Let's set maxiter to something that fits in 60s.
        # With 64 vars, popsize 15, 100 iters is ~ 1500 evals.
        # Each eval is O(N^2) ~ 400 ops. 1500*400 = 600k ops. Should be fast (<1s).
        # So we can afford more iterations.
        
        res_de = differential_evolution(
            lambda x: _calc_penalty_and_obj(x, smooth=False)[0],
            bounds,
            maxiter=200,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=False # We will polish manually
        )
        
        if res_de.success or res_de.x is not None:
            x_de = res_de.x
            # Validate and evaluate
            # DE might return invalid points. We need to project or just use as start for local search.
            # For now, just store as candidate.
            
            # Run Local Refinement from DE result
            # Use smooth objective for local refinement
            def fun_smooth(x):
                return _calc_penalty_and_obj(x, smooth=True)[0]
            
            def grad_smooth(x):
                return _calc_penalty_and_obj(x, smooth=True)[1]
            
            # Time budget for local: remaining time - 5s margin
            local_start = time.time()
            local_deadline = DEADLINE - 5.0
            
            try:
                res_loc = minimize(
                    fun_smooth, x_de, jac=grad_smooth, method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
                )
                if res_loc.x is not None:
                    x_loc = res_loc.x
                    val_loc = _evaluate(x_loc)
                    if val_loc is not None and val_loc > best_val:
                        best_val = val_loc
                        best_x = x_loc
            except Exception:
                pass
                
    except Exception:
        pass

    # If DE failed or didn't find good, try grid init + local
    if best_val < 1.0 or time.time() < DEADLINE - 20:
        # Try a few grid inits
        for W in [0.9, 1.0, 1.1]:
            if time.time() > DEADLINE - 10:
                break
            x0 = _grid_init(W, 2.0-W)
            
            def fun_smooth(x):
                return _calc_penalty_and_obj(x, smooth=True)[0]
            def grad_smooth(x):
                return _calc_penalty_and_obj(x, smooth=True)[1]
                
            try:
                res_loc = minimize(
                    fun_smooth, x0, jac=grad_smooth, method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
                )
                if res_loc.x is not None:
                    val_loc = _evaluate(res_loc.x)
                    if val_loc is not None and val_loc > best_val:
                        best_val = val_loc
                        best_x = res_loc.x
            except Exception:
                pass

    # Fallback
    if best_x is None:
        # Return a simple valid grid
        W = 1.0
        H = 1.0
        x = np.zeros(64)
        x[0] = W
        kx, ky = 5, 5
        cx = np.linspace(0, W, kx+2)[1:-1]
        cy = np.linspace(0, H, ky+2)[1:-1]
        idx = 0
        for yy in cy:
            for xx in cx:
                if idx >= N: break
                r = min(xx, yy, W-xx, H-yy) * 0.9
                x[1+idx] = xx
                x[22+idx] = yy
                x[43+idx] = r
                idx += 1
        best_x = x
        best_val = _evaluate(best_x)
        
    # Format output
    # Return array of shape (N, 3)
    cx = best_x[1:22]
    cy = best_x[22:43]
    r = best_x[43:64]
    
    # Final clamping for safety
    W = best_x[0]
    H = 2.0 - W
    cx = np.clip(cx, 0, W)
    cy = np.clip(cy, 0, H)
    r = np.maximum(r, 1e-6)
    
    # Enforce wall constraints strictly
    for i in range(N):
        max_r = min(cx[i], cy[i], W-cx[i], H-cy[i])
        if r[i] > max_r:
            r[i] = max_r
            
    out = np.column_stack((cx, cy, r))
    
    return out
