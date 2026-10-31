"""Circle packing in perimeter-4 rectangle: Greedy Init + Simulated Annealing + SLSQP Polish."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
_START = None
_DEADLINE = 145.0  # 5s safety margin from 150s limit

def _get_W_H(W):
    H = 2.0 - W
    return W, H

def _calc_penalty_and_obj(x, W, H, mu=1000.0):
    """
    x layout: [cx_0..cx_{N-1}, cy_0..cy_{N-1}, r_0..r_{N-1}]
    Returns (value, grad)
    """
    cx = x[:N]
    cy = x[N:2*N]
    r = x[2*N:3*N]
    
    if W <= 0.1 or H <= 0.1:
        return 1e6, np.zeros_like(x)

    # Objective: -sum(r)
    obj = -np.sum(r)
    grad = np.zeros_like(x)
    grad[2*N:3*N] = -1.0

    # Wall constraints
    d_left = cx
    d_bottom = cy
    d_right = W - cx
    d_top = H - cy
    
    d_min = np.minimum(np.minimum(d_left, d_bottom), np.minimum(d_right, d_top))
    viol_wall = r - d_min
    
    # Smooth penalty (using max(0, v)^2 approximation via softplus or just squared if positive)
    # For SLSQP, hard constraints are better, but we use penalty.
    # Let's use squared violation for smoothness.
    viol_wall_pos = np.maximum(viol_wall, 0.0)
    pen_wall = np.sum(viol_wall_pos**2)
    obj += mu * pen_wall
    
    # Gradient for wall penalty
    for i in range(N):
        if viol_wall_pos[i] > 0:
            # d/dx of (r - d_min)^2 = 2*(r-d_min) * d(r-d_min)/dx
            # d(r-d_min)/dx = -d(d_min)/dx
            # d_min is min of 4 distances.
            min_d = d_min[i]
            g_cx = 0.0
            g_cy = 0.0
            cnt = 0
            if d_left[i] <= min_d + 1e-9:
                g_cx += 1.0
                cnt += 1
            if d_right[i] <= min_d + 1e-9:
                g_cx += -1.0
                cnt += 1
            if d_bottom[i] <= min_d + 1e-9:
                g_cy += 1.0
                cnt += 1
            if d_top[i] <= min_d + 1e-9:
                g_cy += -1.0
                cnt += 1
            if cnt > 0:
                g_cx /= cnt
                g_cy /= cnt
                
            dv = 2.0 * mu * viol_wall_pos[i]
            grad[i] += dv * (-g_cx)
            grad[N+i] += dv * (-g_cy)
            grad[2*N+i] += dv * 1.0

    # Overlap constraints
    for i in range(N):
        for j in range(i+1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.hypot(dx, dy)
            req = r[i] + r[j]
            viol = req - d
            viol_pos = max(viol, 0.0)
            
            if viol_pos > 0:
                obj += mu * viol_pos**2
                if d < 1e-12:
                    dv = 2.0 * mu * viol_pos
                    grad[i] += dv * 1.0
                    grad[j] += dv * -1.0
                    grad[2*N+i] += dv * 1.0
                    grad[2*N+j] += dv * 1.0
                else:
                    dv = 2.0 * mu * viol_pos
                    grad[i] += dv * (-dx / d)
                    grad[N+i] += dv * (-dy / d)
                    grad[j] += dv * (dx / d)
                    grad[N+j] += dv * (dy / d)
                    grad[2*N+i] += dv * 1.0
                    grad[2*N+j] += dv * 1.0

    return obj, grad

def _evaluate(x, W, H):
    """
    Evaluate if x is valid. Return sum of radii or None.
    """
    if W <= 0 or H <= 0:
        return None
        
    cx = x[:N]
    cy = x[N:2*N]
    r = x[2*N:3*N]
    
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

def _greedy_init(W, H):
    """
    Greedy placement: place circles one by one at the position
    that maximizes the minimum distance to existing circles and walls.
    """
    cx = np.zeros(N)
    cy = np.zeros(N)
    r = np.zeros(N)
    
    # First circle at center
    cx[0] = W / 2.0
    cy[0] = H / 2.0
    r[0] = min(W, H) / 2.0 * 0.95
    
    for k in range(1, N):
        best_score = -1e9
        best_x, best_y, best_r = 0, 0, 0
        
        # Search grid for best position
        # Coarse grid first
        for gx in np.linspace(0.05*W, 0.95*W, 10):
            for gy in np.linspace(0.05*H, 0.95*H, 10):
                # Calculate min distance to walls
                d_wall = min(gx, gy, W-gx, H-gy)
                if d_wall < 0.01:
                    continue
                
                # Calculate min distance to existing circles
                min_d = d_wall
                for i in range(k):
                    d = np.hypot(gx - cx[i], gy - cy[i])
                    # Effective distance considering existing radius
                    # We want to place circle k such that it doesn't overlap i.
                    # Distance between centers must be >= r_i + r_k.
                    # We don't know r_k yet.
                    # Let's assume r_k is determined by the closest constraint.
                    # Constraint: dist >= r_i + r_k => r_k <= dist - r_i.
                    # Also r_k <= d_wall.
                    # So max possible r_k at this spot is min(d_wall, min_i(dist_i - r_i)).
                    possible_r = d_wall
                    for i in range(k):
                        d = np.hypot(gx - cx[i], gy - cy[i])
                        possible_r = min(possible_r, d - r[i])
                    
                    if possible_r > 0.01:
                        # Score: we want large radius
                        if possible_r > best_score:
                            best_score = possible_r
                            best_x, best_y, best_r = gx, gy, possible_r
                            
        # If no good spot found, place at center with tiny radius
        if best_score < 0.01:
             best_x, best_y, best_r = W/2, H/2, 0.01
             
        cx[k] = best_x
        cy[k] = best_y
        r[k] = best_r * 0.95 # Shrink slightly for safety
        
    return np.concatenate([cx, cy, r])

def _sa_optimize(x0, W, H, time_limit=10.0):
    """
    Simulated Annealing to improve the initial layout.
    """
    x = x0.copy()
    T = 0.1
    T_min = 1e-4
    alpha = 0.999
    step_size = 0.05
    
    best_x = x.copy()
    best_e = _calc_penalty_and_obj(x, W, H, mu=100.0)[0]
    
    start_t = time.time()
    while time.time() - start_t < time_limit:
        x_new = x.copy()
        noise = np.random.normal(0, step_size * T, size=len(x))
        x_new += noise
        
        # Clamp
        x_new[2*N:3*N] = np.maximum(x_new[2*N:3*N], 1e-4)
        x_new[2*N:3*N] = np.minimum(x_new[2*N:3*N], min(W,H)/2.0)
        x_new[:N] = np.clip(x_new[:N], 0, W)
        x_new[N:2*N] = np.clip(x_new[N:2*N], 0, H)
        
        e_new = _calc_penalty_and_obj(x_new, W, H, mu=100.0)[0]
        
        if e_new < best_e:
            best_e = e_new
            best_x = x_new.copy()
            
        # Acceptance
        e_cur = _calc_penalty_and_obj(x, W, H, mu=100.0)[0]
        if e_new < e_cur or np.random.random() < np.exp(-(e_new - e_cur) / T):
            x = x_new
            
        T = max(T * alpha, T_min)
        
    return best_x

def _slsqp_optimize(x0, W, H, time_limit=5.0):
    """
    Polish the result using SLSQP.
    """
    def fun(x):
        return _calc_penalty_and_obj(x, W, H, mu=1000.0)[0]
        
    def grad(x):
        return _calc_penalty_and_obj(x, W, H, mu=1000.0)[1]
        
    try:
        res = minimize(fun, x0, jac=grad, method='SLSQP',
                       options={'maxiter': 100, 'ftol': 1e-10, 'disp': 0})
        if res.x is not None:
            return res.x
    except Exception:
        pass
    return x0

def construct():
    global _START
    if _START is None:
        _START = time.time()

    best_val = -1e9
    best_result = None
    best_W = 1.0
    
    # Try a few aspect ratios
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    for w in aspect_ratios:
        if time.time() - _START > _DEADLINE - 15:
            break
        h = 2.0 - w
        
        # 1. Greedy Init
        seed = _greedy_init(w, h)
        
        # 2. Simulated Annealing
        x_sa = _sa_optimize(seed, w, h, time_limit=8.0)
        
        # 3. SLSQP Polish
        x_final = _slsqp_optimize(x_sa, w, h, time_limit=5.0)
        
        val = _evaluate(x_final, w, h)
        if val is not None:
            if val > best_val:
                best_val = val
                best_result = (x_final, w)

    # If nothing feasible, return a trivial valid packing
    if best_result is None:
        w = 1.0
        h = 1.0
        r = 1e-4
        xs = np.linspace(r, w - r, N)
        ys = np.full(N, h / 2.0)
        return np.column_stack((xs, ys, r))

    x, w = best_result
    h = 2.0 - w
    cx = x[:N]
    cy = x[N:2*N]
    r = x[2*N:3*N]
    
    # Clamp to be safe
    r = np.maximum(r, 1e-4)
    cx = np.clip(cx, r, w - r)
    cy = np.clip(cy, r, h - r)
    
    # Re-check feasibility after clamping
    if _evaluate(np.concatenate([cx, cy, r]), w, h) is None:
        scale = 0.999
        r = r * scale
        cx = np.clip(cx, r, w - r)
        cy = np.clip(cy, r, h - r)
        
    return np.column_stack((cx, cy, r))
