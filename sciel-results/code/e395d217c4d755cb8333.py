"""Circle packing in perimeter-4 rectangle: Simulated Annealing + Local Search."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _pairwise_distances(x, y):
    """Compute pairwise distances for N circles."""
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    return d


def _is_valid(x, y, r, w, h, tol=1e-8):
    """Check if packing is valid: all circles inside rectangle and non-overlapping."""
    if w <= 0.01 or h <= 0.01:
        return False
    if np.any(x < r - tol) or np.any(x > w - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > h + tol):
        return False
    d = _pairwise_distances(x, y)
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    if np.any(d[iu] < r_sum[iu] - tol):
        return False
    return True


def _objective(x, y, r, w, h, penalty=100.0):
    """Objective function: minimize -sum(r) + penalties."""
    if w < 0.01 or h < 0.01:
        return 1e6
    
    obj = -np.sum(r)
    
    # Pairwise overlap penalty
    d = _pairwise_distances(x, y)
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    overlap = np.maximum(r_sum[iu] - d[iu], 0.0)
    obj += penalty * np.sum(overlap ** 2)
    
    # Boundary penalty
    left_v = np.maximum(r - x, 0.0)
    right_v = np.maximum(x + r - w, 0.0)
    bottom_v = np.maximum(r - y, 0.0)
    top_v = np.maximum(y + r - h, 0.0)
    obj += penalty * (np.sum(left_v ** 2) + np.sum(right_v ** 2) + 
                      np.sum(bottom_v ** 2) + np.sum(top_v ** 2))
    
    return obj


def _anneal_step(x, y, r, w, h, T, rng, penalty=100.0):
    """One step of simulated annealing: propose a move and accept if better or with prob exp(-delta/T)."""
    current_obj = _objective(x, y, r, w, h, penalty)
    
    # Choose random circle and move type
    i = rng.integers(0, N)
    move_type = rng.integers(0, 4)
    
    x_new = x.copy()
    y_new = y.copy()
    r_new = r.copy()
    w_new = w
    h_new = h
    
    if move_type == 0:
        # Move position
        dx = rng.normal(0, 0.05)
        dy = rng.normal(0, 0.05)
        x_new[i] += dx
        y_new[i] += dy
    elif move_type == 1:
        # Change radius
        factor = rng.uniform(0.9, 1.1)
        r_new[i] *= factor
    elif move_type == 2:
        # Change aspect ratio
        dw = rng.normal(0, 0.02)
        w_new = np.clip(w + dw, 0.3, 1.7)
        h_new = 2.0 - w_new
    else:
        # Swap two circles
        j = rng.integers(0, N)
        if j != i:
            x_new[i], x_new[j] = x_new[j], x_new[i]
            y_new[i], y_new[j] = y_new[j], y_new[i]
            r_new[i], r_new[j] = r_new[j], r_new[i]
    
    # Clamp positions to stay roughly in bounds
    for k in range(N):
        x_new[k] = np.clip(x_new[k], 0.01, w_new - 0.01)
        y_new[k] = np.clip(y_new[k], 0.01, h_new - 0.01)
        r_new[k] = max(0.005, min(r_new[k], min(w_new, h_new) * 0.45))
    
    new_obj = _objective(x_new, y_new, r_new, w_new, h_new, penalty)
    
    # Acceptance criterion
    delta = new_obj - current_obj
    if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
        return x_new, y_new, r_new, w_new, new_obj
    
    return x, y, r, w, current_obj


def _local_search(x, y, r, w, h, time_budget, rng, max_rounds=50):
    """Local search: iteratively try to expand each circle and move others to make room."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    w = w.copy() if np.isscalar(w) else w
    h = h.copy() if np.isscalar(h) else h
    
    start_time = time.time()
    
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break
        
        improved = False
        
        for i in range(N):
            if time.time() > time_budget:
                break
            
            # Try to expand circle i
            # Max radius from walls
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            
            # Max radius from other circles
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            
            r_max = max(0.01, r_max)
            
            if r_max > r[i] + 1e-6:
                # Binary search for max valid radius
                lo, hi = r[i], r_max
                for _ in range(15):
                    mid = (lo + hi) / 2
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h):
                        lo = mid
                    else:
                        hi = mid
                r[i] = lo
                improved = True
            
            # If can't expand, try moving circle i
            # Find direction of most constraining neighbor
            min_gap = float('inf')
            best_dx, best_dy = 0, 0
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                gap = d - (r[i] + r[j])
                if gap < min_gap:
                    min_gap = gap
                    if d > 1e-6:
                        best_dx = dx / d
                        best_dy = dy / d
            
            # Also check walls
            d_left = x[i] - r[i]
            d_right = w - x[i] - r[i]
            d_bottom = y[i] - r[i]
            d_top = h - y[i] - r[i]
            
            if d_left < min_gap:
                min_gap = d_left
                best_dx, best_dy = 1, 0
            if d_right < min_gap:
                min_gap = d_right
                best_dx, best_dy = -1, 0
            if d_bottom < min_gap:
                min_gap = d_bottom
                best_dx, best_dy = 0, 1
            if d_top < min_gap:
                min_gap = d_top
                best_dx, best_dy = 0, -1
            
            # Try moving in several directions
            for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                dx_move = np.cos(angle) * 0.02
                dy_move = np.sin(angle) * 0.02
                x_test = x.copy()
                y_test = y.copy()
                x_test[i] += dx_move
                y_test[i] += dy_move
                
                # Clamp
                x_test[i] = np.clip(x_test[i], r[i], w - r[i])
                y_test[i] = np.clip(y_test[i], r[i], h - r[i])
                
                if _is_valid(x_test, y_test, r, w, h):
                    x = x_test
                    y = y_test
                    improved = True
                    break
        
        if not improved:
            break
    
    return x, y, r, w


def _slsqp_refine(x, y, r, w, time_budget):
    """Refine using SLSQP with explicit constraints."""
    n = N
    z = np.zeros(3 * n + 1)
    z[0::3][:n] = x
    z[1::3][:n] = y
    z[2::3][:n] = r
    z[-1] = w
    
    h = 2.0 - w
    
    def obj(zv):
        rv = zv[2::3][:n]
        return -np.sum(rv)
    
    bounds = [(0, 2), (0, 2), (0.001, 1.0)] * n + [(0.3, 1.7)]
    
    constraints = []
    
    # Boundary constraints
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[-1] - p[3*i] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: (2.0 - p[-1]) - p[3*i+1] - p[3*i+2]})
    
    # Overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i, j=j: 
                (p[3*i] - p[3*j])**2 + (p[3*i+1] - p[3*j+1])**2 - (p[3*i+2] + p[3*j+2])**2})
    
    maxiter = int(min(500, time_budget * 100))
    try:
        res = minimize(obj, z, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})
        z_opt = res.x
        x_opt = z_opt[0::3][:n]
        y_opt = z_opt[1::3][:n]
        r_opt = z_opt[2::3][:n]
        w_opt = z_opt[-1]
        h_opt = 2.0 - w_opt
        
        if _is_valid(x_opt, y_opt, r_opt, w_opt, h_opt):
            return x_opt, y_opt, r_opt, w_opt
    except Exception:
        pass
    
    return x, y, r, w


def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    rng = np.random.default_rng(42)
    
    best_value = 0.0
    best_x = None
    best_y = None
    best_r = None
    best_w = None
    
    # Multiple restarts with different initial configurations
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.55, 1.8, 0.5]
    
    for w0 in aspect_ratios:
        if time.time() > deadline - 30:
            break
        
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        
        # Initialize with grid
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        for i in range(N):
            row, col = divmod(i, side)
            x[i] = (col + 0.5) / side * w0
            y[i] = (row + 0.5) / side * h0
            r[i] = min(w0 / side, h0 / side) * 0.4
        
        w = w0
        h = h0
        
        # Simulated annealing
        T_start = 0.1
        T_end = 0.001
        n_steps = 3000
        step_time = (deadline - 30 - (time.time() - start_time)) / (len(aspect_ratios) * n_steps)
        
        for step in range(n_steps):
            if time.time() > deadline - 30:
                break
            
            T = T_start * (T_end / T_start) ** (step / n_steps)
            x, y, r, w, _ = _anneal_step(x, y, r, w, h, T, rng)
            h = 2.0 - w
        
        # Local search
        x, y, r, w = _local_search(x, y, r, w, 2.0 - w, deadline - 25, rng)
        
        # SLSQP refine
        remaining = deadline - (time.time() - start_time) - 10
        if remaining > 5:
            x, y, r, w = _slsqp_refine(x, y, r, w, remaining)
        
        # Check validity and update best
        h = 2.0 - w
        if _is_valid(x, y, r, w, h):
            val = np.sum(r)
            if val > best_value:
                best_value = val
                best_x = x.copy()
                best_y = y.copy()
                best_r = r.copy()
                best_w = w
    
    # Final polish on best result
    if best_x is not None and time.time() < deadline - 5:
        x, y, r, w = _local_search(best_x, best_y, best_r, best_w, 2.0 - best_w, deadline - 3, rng)
        h = 2.0 - w
        if _is_valid(x, y, r, w, h):
            val = np.sum(r)
            if val > best_value:
                best_value = val
                best_x = x.copy()
                best_y = y.copy()
                best_r = r.copy()
                best_w = w
    
    if best_x is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3))
        return np.array(cells[:N])
    
    return np.column_stack((best_x, best_y, best_r))
