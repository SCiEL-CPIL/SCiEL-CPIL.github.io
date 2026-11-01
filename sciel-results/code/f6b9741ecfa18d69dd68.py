"""Circle packing in perimeter-4 rectangle: Differentiable Physics-Based Annealing."""
import numpy as np
import time

N = 21
_TIME_LIMIT = 145.0

def _compute_objective(x, y, r, w, h, beta=100.0, penalty_weight=100.0):
    """
    Compute differentiable objective and gradients.
    Objective: Maximize sum(r) - penalty(overlaps) - penalty(walls)
    """
    x = x.flatten()
    y = y.flatten()
    r = r.flatten()
    
    # 1. Overlap penalties (smooth)
    overlap_val = 0.0
    grad_x = np.zeros(N)
    grad_y = np.zeros(N)
    grad_r = np.zeros(N)
    
    # Pairwise overlaps
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            min_dist = r[i] + r[j]
            dist = np.sqrt(dist_sq + 1e-12)
            
            # Overlap amount
            overlap = min_dist - dist
            if overlap > 0:
                # Smooth penalty: 0.5 * overlap^2
                # Gradient w.r.t x_i, x_j, r_i, r_j
                force = overlap / dist
                overlap_val += 0.5 * overlap * overlap
                
                # d(overlap)/d(x_i) = -dx/dist
                # d(overlap)/d(r_i) = 1
                grad_x[i] -= force * overlap
                grad_y[i] -= force * overlap
                grad_x[j] += force * overlap
                grad_y[j] += force * overlap
                grad_r[i] -= overlap
                grad_r[j] -= overlap
                
    # Wall penalties
    # x_i >= r_i  =>  r_i - x_i <= 0
    # x_i + r_i <= w
    # y_i >= r_i
    # y_i + r_i <= h
    
    wall_val = 0.0
    
    # Left wall: r - x
    viol_l = r - x
    mask = viol_l > 0
    wall_val += np.sum(0.5 * viol_l[mask]**2)
    grad_r[mask] -= viol_l[mask]
    grad_x[mask] += viol_l[mask]
    
    # Right wall: x + r - w
    viol_r = x + r - w
    mask = viol_r > 0
    wall_val += np.sum(0.5 * viol_r[mask]**2)
    grad_r[mask] -= viol_r[mask]
    grad_x[mask] += viol_r[mask]
    
    # Bottom wall: r - y
    viol_b = r - y
    mask = viol_b > 0
    wall_val += np.sum(0.5 * viol_b[mask]**2)
    grad_r[mask] -= viol_b[mask]
    grad_y[mask] += viol_b[mask]
    
    # Top wall: y + r - h
    viol_t = y + r - h
    mask = viol_t > 0
    wall_val += np.sum(0.5 * viol_t[mask]**2)
    grad_r[mask] -= viol_t[mask]
    grad_y[mask] += viol_t[mask]
    
    total_penalty = penalty_weight * (overlap_val + wall_val)
    
    # Objective to MINIMIZE: -sum(r) + total_penalty
    obj_val = -np.sum(r) + total_penalty
    
    # Gradients
    grad_r += penalty_weight * (grad_r) # Wait, grad_r above is from penalty.
    # Let's re-accumulate carefully.
    
    # Actually, let's just return the value and re-compute gradients via finite diff? No, analytic is better.
    # Let's re-do gradient accumulation cleanly.
    
    g_x = np.zeros(N)
    g_y = np.zeros(N)
    g_r = np.zeros(N)
    
    # Objective part: -sum(r) -> grad_r -= 1
    g_r -= 1.0
    
    # Penalty part: penalty_weight * (overlap + wall)
    # Overlap gradients
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            min_dist = r[i] + r[j]
            dist = np.sqrt(dist_sq + 1e-12)
            
            overlap = min_dist - dist
            if overlap > 0:
                force = overlap / dist
                # Penalty term: 0.5 * overlap^2
                # d/dx_i (0.5 * overlap^2) = overlap * d(overlap)/dx_i
                # d(overlap)/dx_i = -dx/dist
                g_x[i] += penalty_weight * overlap * (-dx / dist)
                g_y[i] += penalty_weight * overlap * (-dy / dist)
                g_x[j] += penalty_weight * overlap * (dx / dist)
                g_y[j] += penalty_weight * overlap * (dy / dist)
                g_r[i] += penalty_weight * overlap * (1)
                g_r[j] += penalty_weight * overlap * (1)
                
    # Wall gradients
    # Left: 0.5 * (r-x)^2. d/dr = (r-x), d/dx = -(r-x)
    viol_l = r - x
    mask = viol_l > 0
    g_r[mask] += penalty_weight * viol_l[mask]
    g_x[mask] += penalty_weight * (-viol_l[mask])
    
    # Right: 0.5 * (x+r-w)^2. d/dr = (x+r-w), d/dx = (x+r-w)
    viol_r = x + r - w
    mask = viol_r > 0
    g_r[mask] += penalty_weight * viol_r[mask]
    g_x[mask] += penalty_weight * viol_r[mask]
    
    # Bottom: 0.5 * (r-y)^2. d/dr = (r-y), d/dy = -(r-y)
    viol_b = r - y
    mask = viol_b > 0
    g_r[mask] += penalty_weight * viol_b[mask]
    g_y[mask] += penalty_weight * (-viol_b[mask])
    
    # Top: 0.5 * (y+r-h)^2. d/dr = (y+r-h), d/dy = (y+r-h)
    viol_t = y + r - h
    mask = viol_t > 0
    g_r[mask] += penalty_weight * viol_t[mask]
    g_y[mask] += penalty_weight * viol_t[mask]
    
    # Gradient for w? 
    # The objective depends on w through h = 2-w and wall penalties.
    # But in this simplified differentiable approach, we might fix w and h per run, 
    # or include w in the optimization.
    # Let's fix w and h for now to simplify, and multi-start over aspect ratios.
    # This is safer.
    
    return obj_val, g_x, g_y, g_r

def _is_valid(x, y, r, w, h, tol=1e-9):
    if w <= 0.05 or h <= 0.05:
        return False
    for i in range(len(r)):
        if r[i] <= 0:
            return False
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False
    for i in range(len(r)):
        for j in range(i+1, len(r)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx*dx + dy*dy
            rs = r[i] + r[j]
            if d2 < rs*rs - tol:
                return False
    return True

def _solve_with_adam(x0, y0, r0, w, h, max_iters=2000, lr_start=0.01, lr_end=0.0001, 
                     penalty_weight_start=10.0, penalty_weight_end=1000.0,
                     time_budget=10.0, start_time=None):
    if start_time is None:
        start_time = time.time()
        
    x = x0.copy().astype(np.float64)
    y = y0.copy().astype(np.float64)
    r = r0.copy().astype(np.float64)
    
    # Adam state
    m_x = np.zeros_like(x)
    v_x = np.zeros_like(x)
    m_y = np.zeros_like(y)
    v_y = np.zeros_like(y)
    m_r = np.zeros_like(r)
    v_r = np.zeros_like(r)
    t = 0
    
    best_val = -np.sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    for i in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        t += 1
        # Annealing
        progress = i / max_iters
        lr = lr_start * (lr_end / lr_start) ** progress
        pw = penalty_weight_start * (penalty_weight_end / penalty_weight_start) ** progress
        
        obj_val, g_x, g_y, g_r = _compute_objective(x, y, r, w, h, penalty_weight=pw)
        
        # If valid, update best
        if _is_valid(x, y, r, w, h):
            val = np.sum(r)
            if val > best_val:
                best_val = val
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
                
        # Adam update
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        m_x = beta1 * m_x + (1 - beta1) * g_x
        v_x = beta2 * v_x + (1 - beta2) * g_x**2
        m_x_hat = m_x / (1 - beta1**t)
        v_x_hat = v_x / (1 - beta2**t)
        x -= lr * m_x_hat / (np.sqrt(v_x_hat) + eps)
        
        m_y = beta1 * m_y + (1 - beta1) * g_y
        v_y = beta2 * v_y + (1 - beta2) * g_y**2
        m_y_hat = m_y / (1 - beta1**t)
        v_y_hat = v_y / (1 - beta2**t)
        y -= lr * m_y_hat / (np.sqrt(v_y_hat) + eps)
        
        m_r = beta1 * m_r + (1 - beta1) * g_r
        v_r = beta2 * v_r + (1 - beta2) * g_r**2
        m_r_hat = m_r / (1 - beta1**t)
        v_r_hat = v_r / (1 - beta2**t)
        r -= lr * m_r_hat / (np.sqrt(v_r_hat) + eps)
        
        # Project to valid bounds roughly to help
        x = np.clip(x, 0, w)
        y = np.clip(y, 0, h)
        r = np.maximum(r, 0.01)
        
    return best_x, best_y, best_r, best_val

def _greedy_expand(x, y, r, w, h, max_rounds=50, time_budget=5.0, start_time=None):
    if start_time is None:
        start_time = time.time()
    x = x.copy()
    y = y.copy()
    r = r.copy()
    for _ in range(max_rounds):
        if time.time() - start_time > time_budget:
            break
        improved = False
        for i in range(N):
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)
            if r_max > r[i] + 1e-10:
                r[i] = r_max
                improved = True
        if not improved:
            break
    return x, y, r

def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    
    best_value = 0.0
    best_x = best_y = best_r = best_w = best_h = None
    
    def evaluate(x, y, r, w, h):
        nonlocal best_value, best_x, best_y, best_r, best_w, best_h
        if not _is_valid(x, y, r, w, h):
            return False
        val = float(np.sum(r))
        if val > best_value:
            best_value = val
            best_x, best_y, best_r, best_w, best_h = x.copy(), y.copy(), r.copy(), w, h
            return True
        return False

    # Multi-start over aspect ratios
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8, 1.1, 0.9]
    rng = np.random.default_rng(42)

    for w0 in aspect_ratios:
        if time.time() > deadline - 20:
            break
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue

        # Grid-based initial placement
        cols = int(np.ceil(np.sqrt(N * w0 / h0)))
        rows = int(np.ceil(N / cols))
        cw = w0 / cols
        ch = h0 / rows
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row, col = divmod(i, cols)
            x[i] = (col + 0.5) * cw
            y[i] = (row + 0.5) * ch
            r[i] = min(cw, ch) / 2.0 * 0.8

        # Differentiable Physics-Based Annealing
        x_sol, y_sol, r_sol, val_sol = _solve_with_adam(
            x, y, r, w0, h0, 
            max_iters=3000, 
            lr_start=0.02, 
            lr_end=0.0005,
            penalty_weight_start=5.0, 
            penalty_weight_end=500.0,
            time_budget=10.0, 
            start_time=start_time
        )
        
        # Greedy expand after Adam
        if time.time() < deadline - 10:
            x_e, y_e, r_e = _greedy_expand(x_sol, y_sol, r_sol, w0, h0, max_rounds=50, time_budget=3.0, start_time=start_time)
            evaluate(x_e, y_e, r_e, w0, h0)
        else:
            evaluate(x_sol, y_sol, r_sol, w0, h0)

    # Final polish on best result
    if best_value > 0 and best_x is not None and time.time() < deadline - 5:
        # One more greedy expand
        x_f, y_f, r_f = _greedy_expand(best_x, best_y, best_r, best_w, best_h, max_rounds=50, time_budget=3.0, start_time=start_time)
        evaluate(x_f, y_f, r_f, best_w, best_h)

    # Fallback
    if best_x is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i+0.5)/side, (j+0.5)/side, 0.5/side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Safety shrink
    r_out = best_r.copy()
    for _ in range(10):
        r_trial = r_out * (1 - 1e-10)
        if _is_valid(best_x, best_y, r_trial, best_w, best_h, tol=1e-12):
            r_out = r_trial
        else:
            break

    return np.column_stack((best_x, best_y, r_out))
