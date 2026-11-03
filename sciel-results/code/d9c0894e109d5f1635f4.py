import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _softplus(v, beta=100.0):
    """Smooth approximation of max(0, v)."""
    v = np.asarray(v, dtype=float)
    out = np.where(v > 20, v, np.log1p(np.exp(np.clip(v, -20, 20))))
    return out

def _objective_and_grad(z, w, h, penalty_weight=100.0, smooth_beta=50.0):
    """
    z: array of shape (3*N,) containing [x1, y1, r1, ..., xN, yN, rN]
    Returns: (objective_value, gradient)
    Objective: -sum(r) + penalty_weight * (wall_violations + circle_violations)
    Uses smooth softplus for violations.
    """
    x = z[0::3]
    y = z[1::3]
    r = z[2::3]
    
    obj = -np.sum(r)
    grad = np.zeros_like(z)
    grad[2::3] = -1.0
    
    # Wall violations using smooth approximation
    def smooth_penalty_and_grad(v_vals, beta):
        v = v_vals / beta
        sig = 1.0 / (1.0 + np.exp(-np.clip(v, -50, 50)))
        penalty = np.sum(_softplus(v_vals, beta))
        return penalty, sig
    
    v_x_min = r - x
    v_x_max = r - (w - x)
    v_y_min = r - y
    v_y_max = r - (h - y)
    
    wall_pen = 0.0
    p1, s1 = smooth_penalty_and_grad(v_x_min, smooth_beta)
    wall_pen += p1
    grad[0::3] -= penalty_weight * s1
    grad[2::3] += penalty_weight * s1
    
    p2, s2 = smooth_penalty_and_grad(v_x_max, smooth_beta)
    wall_pen += p2
    grad[0::3] += penalty_weight * s2
    grad[2::3] += penalty_weight * s2
    
    p3, s3 = smooth_penalty_and_grad(v_y_min, smooth_beta)
    wall_pen += p3
    grad[1::3] -= penalty_weight * s3
    grad[2::3] += penalty_weight * s3
    
    p4, s4 = smooth_penalty_and_grad(v_y_max, smooth_beta)
    wall_pen += p4
    grad[1::3] += penalty_weight * s4
    grad[2::3] += penalty_weight * s4
    
    obj += penalty_weight * wall_pen
    
    circle_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d_sq = dx*dx + dy*dy
            d = np.sqrt(d_sq + 1e-16)
            
            r_sum = r[i] + r[j]
            v = r_sum - d
            
            if v > 0.001:
                vb = v / smooth_beta
                sig = 1.0 / (1.0 + np.exp(-np.clip(vb, -50, 50)))
                pen = _softplus(v, smooth_beta)
                circle_pen += pen
                
                coeff = penalty_weight * sig / d
                
                grad[3*i + 0] -= coeff * dx
                grad[3*j + 0] += coeff * dx
                grad[3*i + 1] -= coeff * dy
                grad[3*j + 1] += coeff * dy
                grad[3*i + 2] += penalty_weight * sig
                grad[3*j + 2] += penalty_weight * sig
    
    obj += penalty_weight * circle_pen
    
    return obj, grad

def _is_valid(pos, radii, w, h, tol=1e-7):
    """Check validity of the configuration."""
    # Check bounds
    for i in range(N):
        if pos[i, 0] < -tol or pos[i, 0] > w + tol:
            return False
        if pos[i, 1] < -tol or pos[i, 1] > h + tol:
            return False
        if radii[i] < -tol:
            return False
        # Wall clearance
        if radii[i] > pos[i, 0] + tol:
            return False
        if radii[i] > w - pos[i, 0] + tol:
            return False
        if radii[i] > pos[i, 1] + tol:
            return False
        if radii[i] > h - pos[i, 1] + tol:
            return False
            
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            if d < radii[i] + radii[j] - tol:
                return False
    return True

def _polish(z, w, h, max_iter=500):
    """Final high-precision polish pass with tight tolerances."""
    bounds = []
    r_max = min(w, h) / 2.0
    for i in range(N):
        bounds.append((0.0, w))
        bounds.append((0.0, h))
        bounds.append((0.0, r_max))
    
    try:
        res = minimize(_objective_and_grad, z, args=(w, h, 500.0, 200.0), 
                      method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': max_iter, 'ftol': 1e-18, 'gtol': 1e-14, 'disp': False})
        return res.x
    except Exception:
        return z

def _pack_for_aspect(w, h, start_time, deadline, num_trials=10):
    """
    Pack circles in w x h rectangle using gradient-based optimization.
    """
    best_score = 0
    best_pos = None
    best_radii = None
    
    r_max = min(w, h) / 2.0
    bounds = []
    for i in range(N):
        bounds.append((0.0, w))
        bounds.append((0.0, h))
        bounds.append((0.0, r_max))
        
    for trial in range(num_trials):
        if time.time() > deadline - 15:
            break
            
        if trial == 0:
            cols = int(np.ceil(np.sqrt(N * w / h)))
            rows = int(np.ceil(N / cols))
            if cols * rows < N:
                cols += 1
                rows = int(np.ceil(N / cols))
            
            z0 = np.zeros(3 * N)
            for i in range(N):
                col = i % cols
                row = i // cols
                z0[3*i + 0] = (col + 0.5) * w / cols
                z0[3*i + 1] = (row + 0.5) * h / rows
                cell_w = w / cols
                cell_h = h / rows
                z0[3*i + 2] = min(cell_w, cell_h) / 2.0 * 0.85
        elif trial == 1:
            cols = int(np.ceil(np.sqrt(N * w / h)))
            rows = int(np.ceil(N / cols))
            if cols * rows < N:
                cols += 1
                rows = int(np.ceil(N / cols))
            
            z0 = np.zeros(3 * N)
            for i in range(N):
                col = i % cols
                row = i // cols
                z0[3*i + 0] = (col + 0.5) * w / cols
                z0[3*i + 1] = (row + 0.5) * h / rows
                cell_w = w / cols
                cell_h = h / rows
                z0[3*i + 2] = min(cell_w, cell_h) / 2.0 * 0.7
            z0[0::3] += np.random.randn(N) * 0.05 * min(w, h)
            z0[1::3] += np.random.randn(N) * 0.05 * min(w, h)
            z0[0::3] = np.clip(z0[0::3], 0.01, w - 0.01)
            z0[1::3] = np.clip(z0[1::3], 0.01, h - 0.01)
        else:
            z0 = np.zeros(3 * N)
            z0[0::3] = np.random.uniform(0.05, w - 0.05, N)
            z0[1::3] = np.random.uniform(0.05, h - 0.05, N)
            z0[2::3] = np.random.uniform(0.02, 0.15 * min(w, h), N)
            
        try:
            res = minimize(_objective_and_grad, z0, args=(w, h, 100.0, 50.0), 
                         method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False})
            
            z_opt = res.x
            z_polished = _polish(z_opt, w, h, max_iter=300)
            
            pos = np.column_stack((z_polished[0::3], z_polished[1::3]))
            radii = z_polished[2::3]
            
            if _is_valid(pos, radii, w, h):
                score = np.sum(radii)
                if score > best_score:
                    best_score = score
                    best_pos = pos.copy()
                    best_radii = radii.copy()
        except Exception:
            pass
    
    return best_pos, best_radii, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    np.random.seed(42)
    
    best_global_score = 0
    best_global_result = None
    
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0, 0.45, 2.2]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 20:
            break
        
        w = 2 * ar / (1 + ar)
        h = 2 / (1 + ar)
        
        if ar in [1.0, 0.9, 1.1, 0.8, 1.25]:
            num_trials = 12
        else:
            num_trials = 6
            
        pos, radii, score = _pack_for_aspect(w, h, start_time, deadline, num_trials=num_trials)
        
        if pos is None:
            continue
        
        if score > best_global_score:
            best_global_score = score
            best_global_result = np.column_stack([pos, radii])
    
    if best_global_result is None:
        w = 1.0
        h = 1.0
        side = int(np.ceil(np.sqrt(N)))
        pos = np.zeros((N, 2))
        rad = np.zeros(N)
        r_val = 0.5 / side * 0.8
        for i in range(N):
            col = i % side
            row = i // side
            pos[i, 0] = (col + 0.5) / side
            pos[i, 1] = (row + 0.5) / side
            rad[i] = r_val
        best_global_result = np.column_stack([pos, rad])
    
    return best_global_result
