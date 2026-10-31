import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _is_valid(w, h, x, y, r, tol=1e-9):
    """Check if the packing is valid."""
    if w <= 0 or h <= 0 or w + h > 2.0 + tol:
        return False
    for i in range(N):
        if r[i] < -tol:
            return False
        if x[i] - r[i] < -tol or w - x[i] - r[i] < -tol:
            return False
        if y[i] - r[i] < -tol or h - y[i] - r[i] < -tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx*dx + dy*dy < (r[i] + r[j])**2 - tol:
                return False
    return True

def _penalty_objective(v, beta):
    """
    Smooth penalty-based objective.
    v = [w, h, x1..xN, y1..yN, r1..rN]
    Minimize: -sum(r) + beta * (sum of constraint violations)
    """
    w = v[0]
    h = v[1]
    x = v[2:2+N]
    y = v[2+N:2+2*N]
    r = v[2+2*N:2+3*N]
    
    # Objective: maximize sum of radii
    obj = -np.sum(r)
    
    # Penalty terms
    penalty = 0.0
    
    # Perimeter constraint: w + h <= 2
    if w + h > 2.0:
        penalty += (w + h - 2.0)**2
    
    # Wall constraints
    for i in range(N):
        # Left wall: x_i >= r_i
        viol = max(0, r[i] - x[i])
        penalty += viol**2
        # Right wall: w - x_i >= r_i
        viol = max(0, r[i] - (w - x[i]))
        penalty += viol**2
        # Bottom wall: y_i >= r_i
        viol = max(0, r[i] - y[i])
        penalty += viol**2
        # Top wall: h - y_i >= r_i
        viol = max(0, r[i] - (h - y[i]))
        penalty += viol**2
    
    # Circle-circle constraints
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist2 = dx*dx + dy*dy
            min_dist = r[i] + r[j]
            if dist2 < min_dist**2:
                # Penalty for overlap
                overlap = min_dist**2 - dist2
                penalty += overlap**2
    
    return obj + beta * penalty

def _expand_radii(w, h, x, y, r, max_iters=200):
    """
    Try to expand each circle's radius as much as possible
    while maintaining validity.
    """
    r = r.copy()
    for _ in range(max_iters):
        improved = False
        for i in range(N):
            # Try to increase r[i]
            old_r = r[i]
            # Find max possible radius for circle i
            # Limited by walls
            max_r_wall = min(x[i], w - x[i], y[i], h - y[i])
            # Limited by other circles
            max_r_circles = max_r_wall
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist > 0:
                    # r_i + r_j <= dist => r_i <= dist - r_j
                    max_r_circles = min(max_r_circles, dist - r[j])
            new_r = min(max_r_wall, max_r_circles)
            if new_r > old_r + 1e-10:
                r[i] = new_r
                improved = True
        if not improved:
            break
    return r

def _polish_positions(w, h, x, y, r, time_budget=10.0):
    """
    Polish positions to improve packing.
    Use a smooth optimization to adjust positions.
    """
    start = time.time()
    
    # Variables: x, y only (w, h, r fixed)
    def pos_objective(v):
        x_new = v[:N]
        y_new = v[N:2*N]
        
        # Penalty for overlaps and wall violations
        penalty = 0.0
        
        # Wall constraints
        for i in range(N):
            viol = max(0, r[i] - x_new[i])
            penalty += viol**2
            viol = max(0, r[i] - (w - x_new[i]))
            penalty += viol**2
            viol = max(0, r[i] - y_new[i])
            penalty += viol**2
            viol = max(0, r[i] - (h - y_new[i]))
            penalty += viol**2
        
        # Circle-circle
        for i in range(N):
            for j in range(i+1, N):
                dx = x_new[i] - x_new[j]
                dy = y_new[i] - y_new[j]
                dist2 = dx*dx + dy*dy
                min_dist = r[i] + r[j]
                if dist2 < min_dist**2:
                    penalty += (min_dist**2 - dist2)**2
        
        # Also add a small term to encourage spreading out (avoid clustering)
        # This helps escape local minima
        spread_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x_new[i] - x_new[j]
                dy = y_new[i] - y_new[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 0.01:
                    spread_penalty += (0.01 - dist)**2
        
        return penalty + 0.001 * spread_penalty
    
    v0 = np.concatenate([x, y])
    bounds = []
    for i in range(N):
        bounds.append((max(0, r[i]), min(w, w - r[i])))
    for i in range(N):
        bounds.append((max(0, r[i]), min(h, h - r[i])))
    
    try:
        result = minimize(pos_objective, v0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 1000, 'ftol': 1e-10, 'disp': False})
        if result.success:
            x_new = result.x[:N]
            y_new = result.x[N:2*N]
            if _is_valid(w, h, x_new, y_new, r):
                return x_new, y_new
    except Exception:
        pass
    
    return x, y

def _generate_init(w, h, n, seed=None):
    """Generate a hexagonal-like initial packing."""
    if seed is not None:
        np.random.seed(seed)
    
    r_est = np.sqrt((w * h) / (n * 1.15 * np.pi)) * 0.85
    
    # Try different column counts
    best_xs, best_ys, best_rs = None, None, None
    best_overlap = np.inf
    
    for cols in range(1, int(w / (2*r_est)) + 3):
        if cols == 0:
            continue
        rows = int(np.ceil(n / cols))
        
        # Generate positions
        xs = []
        ys = []
        for i in range(n):
            col = i % cols
            row = i // cols
            x = r_est + col * 2 * r_est
            y = r_est + row * r_est * np.sqrt(3)
            if row % 2 == 1:
                x += r_est
            x = np.clip(x, r_est, w - r_est)
            y = np.clip(y, r_est, h - r_est)
            xs.append(x)
            ys.append(y)
        
        xs = np.array(xs)
        ys = np.array(ys)
        rs = np.full(n, r_est)
        
        # Count overlaps
        overlap = 0
        for i in range(n):
            for j in range(i+1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                if dx*dx + dy*dy < (rs[i] + rs[j])**2:
                    overlap += 1
        
        if overlap < best_overlap:
            best_overlap = overlap
            best_xs, best_ys, best_rs = xs, ys, rs
            if overlap == 0:
                break
    
    if best_xs is None:
        # Fallback: grid
        side = int(np.ceil(np.sqrt(n)))
        best_xs = np.array([((i % side) + 0.5) * (w / side) for i in range(n)])
        best_ys = np.array([((i // side) + 0.5) * (h / side) for i in range(n)])
        best_rs = np.full(n, np.min(w, h) / (2 * side) * 0.8)
    
    return best_xs, best_ys, best_rs

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_value = -np.inf
    best_result = None
    
    np.random.seed(42)
    
    # Try multiple aspect ratios
    aspects = [1.0, 1.2, 0.8, 1.5, 0.67, 1.3, 0.77]
    
    for ar_idx, ar in enumerate(aspects):
        if time.time() > deadline - 15:
            break
        
        w_init = np.sqrt(ar)
        h_init = 1.0 / np.sqrt(ar)
        
        # Generate initial packing
        xs, ys, rs = _generate_init(w_init, h_init, N, seed=42 + ar_idx)
        
        # Phase 1: Optimize positions and radii with penalty method
        # Variables: [w, h, x1..xN, y1..yN, r1..rN]
        v0 = np.concatenate([[w_init, h_init], xs, ys, rs])
        
        # Adaptive beta: start small, increase if constraints violated
        beta = 1.0
        v_current = v0.copy()
        
        for beta_iter in range(5):
            # Clip to valid range
            v_current[0] = np.clip(v_current[0], 0.1, 1.9)
            v_current[1] = np.clip(v_current[1], 0.1, 1.9)
            for k in range(N):
                v_current[2+k] = np.clip(v_current[2+k], 0.01, 1.99)
                v_current[2+N+k] = np.clip(v_current[2+N+k], 0.01, 1.99)
                v_current[2+2*N+k] = np.clip(v_current[2+2*N+k], 0.01, 0.99)
            
            try:
                result = minimize(
                    lambda v: _penalty_objective(v, beta),
                    v_current,
                    method='L-BFGS-B',
                    bounds=[(0.1, 1.9), (0.1, 1.9)] + 
                           [(0.01, 1.99)]*N + [(0.01, 1.99)]*N + [(0.01, 0.99)]*N,
                    options={'maxiter': 500, 'ftol': 1e-10, 'disp': False}
                )
                v_current = result.x
            except Exception:
                break
            
            # Check constraint violations
            w = v_current[0]
            h = v_current[1]
            x = v_current[2:2+N]
            y = v_current[2+N:2+2*N]
            r = v_current[2+2*N:2+3*N]
            
            max_viol = 0
            if w + h > 2.0:
                max_viol = max(max_viol, w + h - 2.0)
            for i in range(N):
                max_viol = max(max_viol, max(0, r[i] - x[i]))
                max_viol = max(max_viol, max(0, r[i] - (w - x[i])))
                max_viol = max(max_viol, max(0, r[i] - y[i]))
                max_viol = max(max_viol, max(0, r[i] - (h - y[i])))
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist2 = dx*dx + dy*dy
                    min_dist = r[i] + r[j]
                    if dist2 < min_dist**2:
                        max_viol = max(max_viol, np.sqrt(min_dist**2) - np.sqrt(dist2))
            
            if max_viol < 1e-4:
                break
            else:
                beta *= 5.0
        
        # Extract result
        w = v_current[0]
        h = v_current[1]
        x = v_current[2:2+N].copy()
        y = v_current[2+N:2+2*N].copy()
        r = v_current[2+2*N:2+3*N].copy()
        
        # Phase 2: Polish positions
        if time.time() < deadline - 10:
            x, y = _polish_positions(w, h, x, y, r, time_budget=8.0)
        
        # Phase 3: Expand radii
        if time.time() < deadline - 5:
            r = _expand_radii(w, h, x, y, r, max_iters=100)
        
        # Verify validity
        if _is_valid(w, h, x, y, r, tol=1e-7):
            value = np.sum(r)
            if value > best_value:
                best_value = value
                best_result = (w, h, x.copy(), y.copy(), r.copy())
        
        # Also try SLSQP with constraints for refinement if time allows
        if time.time() < deadline - 15 and best_result is not None:
            w_best, h_best, x_best, y_best, r_best = best_result
            v_sl = np.concatenate([[w_best, h_best], x_best, y_best, r_best])
            
            def sl_objective(v):
                return -np.sum(v[2+2*N:2+3*N])
            
            cons = []
            # Perimeter
            cons.append({'type': 'ineq', 'fun': lambda v: 2.0 - v[0] - v[1]})
            # Walls
            for i in range(N):
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2+i] - v[2+2*N+i]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[0] - v[2+i] - v[2+2*N+i]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2+N+i] - v[2+2*N+i]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[1] - v[2+N+i] - v[2+2*N+i]})
            # Circles
            for i in range(N):
                for j in range(i+1, N):
                    cons.append({
                        'type': 'ineq',
                        'fun': lambda v, i=i, j=j: (v[2+i] - v[2+j])**2 + (v[2+N+i] - v[2+N+j])**2 - (v[2+2*N+i] + v[2+2*N+j])**2
                    })
            
            sl_bounds = [(0.1, 1.9), (0.1, 1.9)] + \
                       [(0.01, 1.99)]*N + [(0.01, 1.99)]*N + [(0.01, 0.99)]*N
            
            try:
                sl_result = minimize(
                    sl_objective, v_sl, method='SLSQP',
                    bounds=sl_bounds, constraints=cons,
                    options={'maxiter': 300, 'ftol': 1e-9, 'disp': False}
                )
                if sl_result.success:
                    w_sl = sl_result.x[0]
                    h_sl = sl_result.x[1]
                    x_sl = sl_result.x[2:2+N]
                    y_sl = sl_result.x[2+N:2+2*N]
                    r_sl = sl_result.x[2+2*N:2+3*N]
                    if _is_valid(w_sl, h_sl, x_sl, y_sl, r_sl, tol=1e-7):
                        value_sl = np.sum(r_sl)
                        if value_sl > best_value:
                            best_value = value_sl
                            best_result = (w_sl, h_sl, x_sl.copy(), y_sl.copy(), r_sl.copy())
            except Exception:
                pass
    
    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        w = 1.0
        h = 1.0
        xs = np.array([((i % side) + 0.5) / side for i in range(N)])
        ys = np.array([((i // side) + 0.5) / side for i in range(N)])
        rs = np.full(N, 0.5 / side - 0.01)
        return np.column_stack([xs, ys, rs])
    else:
        w, h, x, y, r = best_result
        return np.column_stack([x, y, r])
