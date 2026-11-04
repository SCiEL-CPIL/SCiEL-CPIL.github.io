import numpy as np
import time
from scipy.optimize import minimize

N = 26
_DEADLINE_MARGIN = 10.0  # seconds before hard limit

def _elapsed(start):
    return time.time() - start

def _compute_radii(x, y):
    """Compute max radii for centers x, y such that circles are disjoint and inside [0,1]^2."""
    r_bound = np.minimum(
        np.minimum(x, 1.0 - x),
        np.minimum(y, 1.0 - y)
    )
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist, axis=1)
    r = np.minimum(r_bound, 0.5 * min_dist)
    r = np.maximum(r, 1e-6)
    return r

def _smooth_objective_and_grad(params, beta=50.0):
    """
    params: (2N,) vector [x0, y0, x1, y1, ...]
    Returns (-sum(r), grad) where r_i = softmin of constraints.
    """
    x = params[:N]
    y = params[N:2*N]
    
    # Boundary constraints
    r_bound = np.minimum(
        np.minimum(x, 1.0 - x),
        np.minimum(y, 1.0 - y)
    )
    
    # Pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2 + 1e-30)
    np.fill_diagonal(dist, 1e10)
    
    # Constraint matrix: c_ij = 0.5*dist(i,j) for i!=j, c_ii = r_bound[i]
    c = 0.5 * dist
    np.fill_diagonal(c, r_bound)
    
    # Soft-min: r_i = softmin_j(c_ij)
    neg_beta_c = -beta * c
    max_neg = np.max(neg_beta_c, axis=1, keepdims=True)
    exp_terms = np.exp(neg_beta_c - max_neg)
    sum_exp = np.sum(exp_terms, axis=1, keepdims=True)
    log_sum = np.log(sum_exp)
    r = -(max_neg + log_sum) / beta
    r = r[:, 0]
    
    # Weights
    w = exp_terms / sum_exp  # (N, N)
    
    # Gradient of c_ij w.r.t. parameters
    dist_safe = dist.copy()
    np.fill_diagonal(dist_safe, 1.0)
    
    # Off-diagonal gradients
    dc_dx_i = 0.5 * dx / dist_safe
    dc_dy_i = 0.5 * dy / dist_safe
    dc_dx_j = -0.5 * dx / dist_safe
    dc_dy_j = -0.5 * dy / dist_safe
    
    # Diagonal gradients for r_bound_i using smooth approximation
    bounds_4 = np.stack([x, 1.0-x, y, 1.0-y], axis=1)
    beta_b = 100.0
    neg_bb = -beta_b * bounds_4
    max_bb = np.max(neg_bb, axis=1, keepdims=True)
    exp_bb = np.exp(neg_bb - max_bb)
    sum_bb = np.sum(exp_bb, axis=1, keepdims=True)
    w_b = exp_bb / sum_bb
    
    grad_rb_dx = w_b[:, 0] - w_b[:, 1]
    grad_rb_dy = w_b[:, 2] - w_b[:, 3]
    
    np.fill_diagonal(dc_dx_i, grad_rb_dx)
    np.fill_diagonal(dc_dy_i, grad_rb_dy)
    np.fill_diagonal(dc_dx_j, grad_rb_dx)
    np.fill_diagonal(dc_dy_j, grad_rb_dy)
    
    # Total gradient
    dr_dx_self = np.sum(w * dc_dx_i, axis=1)
    dr_dy_self = np.sum(w * dc_dy_i, axis=1)
    dr_dx_cross = np.sum(w * dc_dx_j, axis=0)
    dr_dy_cross = np.sum(w * dc_dy_j, axis=0)
    
    dr_dx = dr_dx_self + dr_dx_cross - np.diag(w) * np.diag(dc_dx_i)
    dr_dy = dr_dy_self + dr_dy_cross - np.diag(w) * np.diag(dc_dy_i)
    
    grad = np.concatenate([-dr_dx, -dr_dy])
    obj_val = -np.sum(r)
    return obj_val, grad

def _check_validity(x, y, r, tol=1e-9):
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    if np.any(r < -tol):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = r[:, None] + r[None, :]
    np.fill_diagonal(dist, np.inf)
    np.fill_diagonal(min_dist, 0)
    if np.any(dist < min_dist - tol):
        return False
    return True

def _score(x, y, r):
    if _check_validity(x, y, r):
        return np.sum(r)
    for scale in [0.999, 0.99, 0.98, 0.95, 0.90]:
        r_test = r * scale
        if _check_validity(x, y, r_test):
            return np.sum(r_test)
    return -np.inf

def _grow_radii(x, y, r, max_iters=200, start_time=0, deadline=100):
    """Greedy radius growth."""
    r = r.copy()
    for it in range(max_iters):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            dr = 0.002
            r_new = r.copy()
            r_new[i] += dr
            if r_new[i] > 0.5:
                continue
            if x[i] - r_new[i] < -1e-10 or x[i] + r_new[i] > 1 + 1e-10:
                continue
            if y[i] - r_new[i] < -1e-10 or y[i] + r_new[i] > 1 + 1e-10:
                continue
            valid = True
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < r_new[i] + r_new[j] - 1e-10:
                    valid = False
                    break
            if valid:
                r[i] = r_new[i]
                improved = True
        if not improved:
            break
    return r

def _gradient_polish(x, y, max_iters=300, start_time=0, deadline=100):
    """L-BFGS-B polish with iterative beta sharpening."""
    x = x.copy()
    y = y.copy()
    best_score = _score(x, y, _compute_radii(x, y))
    
    for beta in [10, 50, 100, 200, 500, 1000]:
        if time.time() > deadline:
            break
        params = np.concatenate([x, y])
        
        def obj_and_grad(p):
            xx = p[:N]
            yy = p[N:]
            return _smooth_objective_and_grad(p, beta)
        
        try:
            res = minimize(obj_and_grad, params, method='L-BFGS-B', 
                          options={'maxiter': 800, 'gtol': 1e-8})
            x_new = res.x[:N]
            y_new = res.x[N:]
            
            # Clip to [0,1]
            x_new = np.clip(x_new, 0.01, 0.99)
            y_new = np.clip(y_new, 0.01, 0.99)
            
            r_new = _compute_radii(x_new, y_new)
            s_new = _score(x_new, y_new, r_new)
            
            if s_new > best_score:
                x, y = x_new, y_new
                best_score = s_new
        except Exception:
            pass
    
    return x, y, _compute_radii(x, y), best_score

def _high_precision_polish(x, y, r, deadline=100):
    """Iterative local refinement: move each circle to maximize its radius."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for iteration in range(30):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            best_pos = (x[i], y[i])
            best_r = r[i]
            
            # Fine grid around current position
            for dx in np.linspace(-0.02, 0.02, 9):
                for dy in np.linspace(-0.02, 0.02, 9):
                    cx = x[i] + dx
                    cy = y[i] + dy
                    if cx < 0.001 or cx > 0.999 or cy < 0.001 or cy > 0.999:
                        continue
                    
                    max_r = min(cx, 1 - cx, cy, 1 - cy)
                    for j in range(N):
                        if j == i:
                            continue
                        ddx = cx - x[j]
                        ddy = cy - y[j]
                        dist = np.sqrt(ddx*ddx + ddy*ddy)
                        if dist > 0:
                            max_r = min(max_r, dist - r[j])
                    max_r = max(0.0, max_r)
                    
                    if max_r > best_r + 1e-12:
                        best_r = max_r
                        best_pos = (cx, cy)
            
            if best_pos[0] != x[i] or best_pos[1] != y[i] or best_r > r[i] + 1e-12:
                x[i] = best_pos[0]
                y[i] = best_pos[1]
                r[i] = best_r
                improved = True
        
        if not improved:
            break
    
    return x, y, r, np.sum(r)

def _best_fit_local_search(x, y, r, max_iters=500, start_time=0, deadline=100):
    """Iteratively move each circle to maximize its radius given others fixed."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            best_pos = (x[i], y[i])
            best_r = r[i]
            
            # Random perturbations
            for _ in range(20):
                angle = np.random.uniform(0, 2 * np.pi)
                dist = np.random.uniform(0, 0.08)
                nx = x[i] + dist * np.cos(angle)
                ny = y[i] + dist * np.sin(angle)
                if 0.001 <= nx <= 0.999 and 0.001 <= ny <= 0.999:
                    max_r = min(nx, 1 - nx, ny, 1 - ny)
                    for j in range(N):
                        if j == i:
                            continue
                        dx = nx - x[j]
                        dy = ny - y[j]
                        d = np.sqrt(dx*dx + dy*dy)
                        if d > 0:
                            max_r = min(max_r, d - r[j])
                    max_r = max(0.0, max_r)
                    
                    if max_r > best_r + 1e-10:
                        best_r = max_r
                        best_pos = (nx, ny)
                        improved = True
            
            # Fine grid refinement
            if improved:
                fine_best_r = best_r
                fine_best_pos = best_pos
                for fdx in np.linspace(-0.015, 0.015, 7):
                    for fdy in np.linspace(-0.015, 0.015, 7):
                        fx = best_pos[0] + fdx
                        fy = best_pos[1] + fdy
                        if fx < 0.001 or fx > 0.999 or fy < 0.001 or fy > 0.999:
                            continue
                        max_r = min(fx, 1 - fx, fy, 1 - fy)
                        for j in range(N):
                            if j == i:
                                continue
                            dx = fx - x[j]
                            dy = fy - y[j]
                            d = np.sqrt(dx*dx + dy*dy)
                            if d > 0:
                                max_r = min(max_r, d - r[j])
                        max_r = max(0.0, max_r)
                        
                        if max_r > fine_best_r + 1e-12:
                            fine_best_r = max_r
                            fine_best_pos = (fx, fy)
                
                best_r = fine_best_r
                best_pos = fine_best_pos
            
            if best_r > r[i] + 1e-10:
                x[i] = best_pos[0]
                y[i] = best_pos[1]
                r[i] = best_r
    
    return x, y, r, np.sum(r)

def _grid_init(seed=42):
    """Grid initial configuration."""
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(N)))
    x = np.zeros(N)
    y = np.zeros(N)
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            x[idx] = (i + 0.5) / side
            y[idx] = (j + 0.5) / side
            idx += 1
    x += rng.uniform(-0.08, 0.08, N)
    y += rng.uniform(-0.08, 0.08, N)
    x = np.clip(x, 0.02, 0.98)
    y = np.clip(y, 0.02, 0.98)
    r = _compute_radii(x, y)
    return x, y, r

def _hex_init(seed=42):
    """Hexagonal initial configuration."""
    rng = np.random.RandomState(seed)
    x = np.zeros(N)
    y = np.zeros(N)
    idx = 0
    side = int(np.ceil(np.sqrt(N)))
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            x[idx] = (i + 0.5) / side + (0.5 / side) * (j % 2)
            y[idx] = (j + 0.5) / side
            idx += 1
    x += rng.uniform(-0.05, 0.05, N)
    y += rng.uniform(-0.05, 0.05, N)
    x = np.clip(x, 0.02, 0.98)
    y = np.clip(y, 0.02, 0.98)
    r = _compute_radii(x, y)
    return x, y, r

def _random_init(seed=42):
    """Random initial configuration."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.05, 0.95, N)
    y = rng.uniform(0.05, 0.95, N)
    r = _compute_radii(x, y)
    return x, y, r

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - _DEADLINE_MARGIN  # 140s safety
    
    best_score = -np.inf
    best_x = None
    best_y = None
    best_r = None
    
    # Generate multiple initial configurations
    inits = []
    
    # Grid init
    x, y, r = _grid_init(42)
    inits.append((x, y, r))
    
    # Hex init
    x, y, r = _hex_init(42)
    inits.append((x, y, r))
    
    # Random inits
    for seed in [123, 456, 789, 1011, 2020]:
        x, y, r = _random_init(seed)
        inits.append((x, y, r))
    
    # Run optimization pipeline on each
    for x_init, y_init, r_init in inits:
        if time.time() > deadline:
            break
        
        try:
            # 1. Best-fit local search
            x1, y1, r1, s1 = _best_fit_local_search(x_init, y_init, r_init, 
                                                     max_iters=300, 
                                                     start_time=start_time, 
                                                     deadline=deadline)
            
            # 2. Gradient polish with L-BFGS-B
            x2, y2, r2, s2 = _gradient_polish(x1, y1, 
                                               max_iters=200, 
                                               start_time=start_time, 
                                               deadline=deadline)
            
            # 3. Grow radii
            r3 = _grow_radii(x2, y2, r2, 
                            max_iters=100, 
                            start_time=start_time, 
                            deadline=deadline)
            
            # 4. High-precision polish
            x4, y4, r4, s4 = _high_precision_polish(x2, y2, r3, 
                                                    deadline=deadline)
            
            # Validate
            if _check_validity(x4, y4, r4):
                score = np.sum(r4)
                if score > best_score:
                    best_score = score
                    best_x = x4.copy()
                    best_y = y4.copy()
                    best_r = r4.copy()
        except Exception:
            pass
    
    # If no valid solution found, use a simple grid fallback
    if best_x is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells)
    
    return np.column_stack((best_x, best_y, best_r))
