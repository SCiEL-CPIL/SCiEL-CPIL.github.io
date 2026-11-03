import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _objective_and_grad(x, w, h, n_circles, penalty_factor):
    """
    Objective: minimize -sum(r) + penalty
    x = [x1, y1, r1, ..., xn, yn, rn]
    """
    x_coords = x[0:n_circles*2:2]
    y_coords = x[1:n_circles*2:2]
    r_coords = x[2:n_circles*2:2]
    
    val = -np.sum(r_coords)
    
    grad = np.zeros_like(x)
    grad[2:n_circles*2:2] = -1.0
    
    for i in range(n_circles):
        for j in range(i+1, n_circles):
            dx = x_coords[i] - x_coords[j]
            dy = y_coords[i] - y_coords[j]
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist < 1e-8:
                dist = 1e-8
                dx = 1e-8
                dy = 0.0
            
            sum_r = r_coords[i] + r_coords[j]
            violation = sum_r - dist
            
            if violation > 0:
                val += penalty_factor * violation**2
                g_penalty = 2.0 * penalty_factor * violation
                grad[2*i] += -g_penalty * (dx / dist)
                grad[2*i+1] += -g_penalty * (dy / dist)
                grad[2*i+2] += g_penalty
                grad[2*j] += g_penalty * (dx / dist)
                grad[2*j+1] += g_penalty * (dy / dist)
                grad[2*j+2] += g_penalty
    
    for i in range(n_circles):
        r = r_coords[i]
        x = x_coords[i]
        y = y_coords[i]
        
        v = r - x
        if v > 0:
            val += penalty_factor * v**2
            grad[2*i] += -2.0 * penalty_factor * v
            grad[2*i+2] += 2.0 * penalty_factor * v
            
        v = r - (w - x)
        if v > 0:
            val += penalty_factor * v**2
            grad[2*i] += 2.0 * penalty_factor * v
            grad[2*i+2] += 2.0 * penalty_factor * v
            
        v = r - y
        if v > 0:
            val += penalty_factor * v**2
            grad[2*i+1] += -2.0 * penalty_factor * v
            grad[2*i+2] += 2.0 * penalty_factor * v
            
        v = r - (h - y)
        if v > 0:
            val += penalty_factor * v**2
            grad[2*i+1] += 2.0 * penalty_factor * v
            grad[2*i+2] += 2.0 * penalty_factor * v
            
    return val, grad

def _validate_solution(pos_r, w, h, n_circles, tol=1e-5):
    for i in range(n_circles):
        xi = pos_r[2*i]
        yi = pos_r[2*i+1]
        ri = pos_r[2*i+2]
        
        if xi < -tol or xi > w + tol or yi < -tol or yi > h + tol or ri < -tol:
            return False
        if ri - xi < -tol or ri - (w-xi) < -tol or ri - yi < -tol or ri - (h-yi) < -tol:
            return False
            
    for i in range(n_circles):
        for j in range(i+1, n_circles):
            xi = pos_r[2*i]
            yi = pos_r[2*i+1]
            ri = pos_r[2*i+2]
            xj = pos_r[2*j]
            yj = pos_r[2*j+1]
            rj = pos_r[2*j+2]
            
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            if dist < (ri + rj) - tol:
                return False
    return True

def _fix_solution(pos_r, w, h, n_circles, tol=1e-5):
    """
    Attempt to fix small constraint violations by shrinking radii.
    """
    # Check wall violations and shrink radii if needed
    for i in range(n_circles):
        xi = pos_r[2*i]
        yi = pos_r[2*i+1]
        ri = pos_r[2*i+2]
        
        # Distance to walls
        d_left = xi
        d_right = w - xi
        d_bottom = yi
        d_top = h - yi
        
        min_wall_dist = min(d_left, d_right, d_bottom, d_top)
        
        if ri > min_wall_dist + tol:
            pos_r[2*i+2] = max(0, min_wall_dist - tol)
            
    # Check circle-circle overlaps and shrink radii
    for _ in range(100):
        fixed = False
        for i in range(n_circles):
            for j in range(i+1, n_circles):
                xi = pos_r[2*i]
                yi = pos_r[2*i+1]
                ri = pos_r[2*i+2]
                xj = pos_r[2*j]
                yj = pos_r[2*j+1]
                rj = pos_r[2*j+2]
                
                dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                required = ri + rj
                
                if dist < required - tol:
                    # Shrink the larger circle
                    if ri >= rj:
                        excess = required - dist
                        pos_r[2*i+2] = max(0, ri - excess - tol)
                    else:
                        excess = required - dist
                        pos_r[2*j+2] = max(0, rj - excess - tol)
                    fixed = True
                    
        if not fixed:
            break
            
    return pos_r

def _pack_for_aspect(w, h, num_starts=3, time_limit=40):
    n_circles = N
    n_vars = n_circles * 3
    max_r = min(w, h) / 2.0
    
    bounds = []
    for i in range(n_circles):
        bounds.append((0, w))
        bounds.append((0, h))
        bounds.append((0, max_r))
        
    best_val = np.inf
    best_x = None
    
    start_time = time.time()
    
    for start_idx in range(num_starts):
        if time.time() - start_time > time_limit:
            break
        
        # Staggered grid initialization
        cols = int(np.ceil(np.sqrt(n_circles * w / h)))
        rows = int(np.ceil(n_circles / cols))
        
        x0 = np.zeros(n_vars)
        for i in range(n_circles):
            col = i % cols
            row = i // cols
            
            # Staggered positions
            if row % 2 == 0:
                cx = (col + 0.5) * w / cols
            else:
                cx = (col + 1.0) * w / cols
                if cx > w:
                    cx = (col - 0.5) * w / cols
            
            cy = (row + 0.5) * h / rows
            
            # Add noise
            noise_x = 0.1 * w / cols
            noise_y = 0.1 * h / rows
            x0[2*i] = cx + np.random.uniform(-noise_x, noise_x)
            x0[2*i+1] = cy + np.random.uniform(-noise_y, noise_y)
            x0[2*i+2] = 0.3 * min(w/cols, h/rows)
            
        for i in range(n_circles):
            x0[2*i] = np.clip(x0[2*i], 0, w)
            x0[2*i+1] = np.clip(x0[2*i+1], 0, h)
            x0[2*i+2] = np.clip(x0[2*i+2], 0, max_r)
            
        penalty = 100.0
        
        def callback(Xk):
            return time.time() - start_time > time_limit
        
        try:
            res = minimize(
                lambda x: _objective_and_grad(x, w, h, n_circles, penalty)[0],
                x0,
                jac=lambda x: _objective_and_grad(x, w, h, n_circles, penalty)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-12, 'maxfun': 10000, 'callback': callback}
            )
            
            pos_r = res.x.copy()
            
            # Validate and fix
            if _validate_solution(pos_r, w, h, n_circles):
                val = -np.sum(pos_r[2::2])
                if val < best_val:
                    best_val = val
                    best_x = pos_r
            else:
                fixed_r = _fix_solution(pos_r, w, h, n_circles)
                if _validate_solution(fixed_r, w, h, n_circles):
                    val = -np.sum(fixed_r[2::2])
                    if val < best_val:
                        best_val = val
                        best_x = fixed_r
                        
        except Exception:
            pass

    if best_x is None:
        return None, 0.0
        
    positions = np.column_stack([best_x[0::2], best_x[1::2]])
    radii = best_x[2::2]
    radii = np.maximum(radii, 0)
    
    score = np.sum(radii)
    
    return np.column_stack([positions, radii]), score

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_result = None
    best_score = 0.0
    
    aspect_configs = [
        (1.0, 1.0),
        (1.2, 0.8),
        (0.8, 1.2),
        (1.4, 0.6),
        (0.6, 1.4),
        (1.5, 0.5)
    ]
    
    for w, h in aspect_configs:
        if time.time() > deadline:
            break
            
        time_remaining = deadline - time.time()
        if time_remaining < 10:
            continue
            
        num_starts = 3 if time_remaining > 60 else 1
        
        result, score = _pack_for_aspect(w, h, num_starts=num_starts, time_limit=time_remaining - 5)
        
        if result is not None and score > best_score:
            best_score = score
            best_result = result
            
    if best_result is None:
        w, h = 1.0, 1.0
        cols = 5
        rows = 5
        positions = []
        radii = []
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N:
                    break
                pos = [(i + 0.5) * w / cols, (j + 0.5) * h / rows]
                r = 0.5 * min(w/cols, h/rows) * 0.95
                positions.append(pos)
                radii.append(r)
                count += 1
        positions = np.array(positions)
        radii = np.array(radii)
        best_result = np.column_stack([positions, radii])
        
    return best_result
