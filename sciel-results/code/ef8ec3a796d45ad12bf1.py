import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _check_validity(x, y, r, W, H, tol=1e-10):
    """Check if the packing is valid."""
    if W <= 0 or H <= 0:
        return False
    if abs(W + H - 2.0) > 1e-8:
        return False
    for i in range(N):
        if r[i] < 0:
            return False
        if x[i] < r[i] - tol or x[i] > W - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > H - r[i] + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _separate_circles(x, y, r, W, H, max_iters=200, tol=1e-12):
    """Iteratively separate overlapping circles and fix boundary violations."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for _ in range(max_iters):
        moved = False
        # Fix boundary violations
        for i in range(N):
            if x[i] < r[i]:
                x[i] = r[i]
                moved = True
            elif x[i] > W - r[i]:
                x[i] = W - r[i]
                moved = True
            if y[i] < r[i]:
                y[i] = r[i]
                moved = True
            elif y[i] > H - r[i]:
                y[i] = H - r[i]
                moved = True
        
        # Separate overlaps
        for i in range(N):
            for j in range(i+1, N):
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    if dist < 1e-15:
                        dx, dy = 1.0, 0.0
                        dist = 1.0
                    overlap = min_dist - dist
                    # Push apart along the line connecting centers
                    factor = 0.5 * overlap / dist
                    x[i] -= factor * dx
                    y[i] -= factor * dy
                    x[j] += factor * dx
                    y[j] += factor * dy
                    moved = True
        
        if not moved:
            break
    
    return x, y, r

def _objective_and_grad(v, mu=100.0):
    """Compute objective and gradient for penalty-based optimization."""
    x = v[0:N]
    y = v[N:2*N]
    r = v[2*N:3*N]
    W = v[3*N]
    H = 2.0 - W
    
    # Objective: -sum(r) + penalty
    obj = -np.sum(r)
    grad = np.zeros_like(v)
    grad[2*N:3*N] = -1.0  # gradient of -sum(r)
    
    # Boundary penalty
    # x in [r, W-r], y in [r, H-r]
    # Penalty: mu * sum of violations^2
    
    # Left boundary: max(0, r_i - x_i)
    viol_left = np.maximum(0, r - x)
    obj += mu * np.sum(viol_left**2)
    grad[0:N] += -2*mu*viol_left
    grad[2*N:3*N] += 2*mu*viol_left
    
    # Right boundary: max(0, x_i + r_i - W)
    viol_right = np.maximum(0, x + r - W)
    obj += mu * np.sum(viol_right**2)
    grad[0:N] += 2*mu*viol_right
    grad[2*N:3*N] += 2*mu*viol_right
    grad[3*N] += -2*mu*np.sum(viol_right)  # dW term
    
    # Bottom boundary: max(0, r_i - y_i)
    viol_bottom = np.maximum(0, r - y)
    obj += mu * np.sum(viol_bottom**2)
    grad[N:2*N] += -2*mu*viol_bottom
    grad[2*N:3*N] += 2*mu*viol_bottom
    
    # Top boundary: max(0, y_i + r_i - H) = max(0, y_i + r_i - (2-W))
    viol_top = np.maximum(0, y + r - (2.0 - W))
    obj += mu * np.sum(viol_top**2)
    grad[N:2*N] += 2*mu*viol_top
    grad[2*N:3*N] += 2*mu*viol_top
    grad[3*N] += 2*mu*np.sum(viol_top)  # dW term (H = 2-W, so dH/dW = -1)
    
    # Overlap penalty: mu * sum over pairs of max(0, r_i + r_j - dist_ij)^2
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r[i] + r[j]
            overlap = max(0, min_dist - dist)
            if overlap > 0:
                obj += mu * overlap**2
                # Gradient of overlap^2
                # d(overlap^2)/d(x_i) = 2*overlap * d(overlap)/d(x_i)
                # d(overlap)/d(x_i) = -dx/dist (if dist > 0)
                if dist > 1e-15:
                    gx_i = -2*mu*overlap*dx/dist
                    gy_i = -2*mu*overlap*dy/dist
                    gx_j = 2*mu*overlap*dx/dist
                    gy_j = 2*mu*overlap*dy/dist
                else:
                    gx_i = gy_i = gx_j = gy_j = 0
                grad[0:N][i] += gx_i
                grad[N:2*N][i] += gy_i
                grad[0:N][j] += gx_j
                grad[N:2*N][j] += gy_j
                grad[2*N:3*N][i] += 2*mu*overlap
                grad[2*N:3*N][j] += 2*mu*overlap
    
    return obj, grad

def _local_optimize(v0, max_iter=100, tol=1e-10, mu_start=10.0, mu_factor=2.0, max_mu=10000.0):
    """Local optimization using penalty method with increasing penalty."""
    v = v0.copy()
    mu = mu_start
    
    for _ in range(max_iter):
        # Run L-BFGS-B with current penalty
        try:
            res = minimize(_objective_and_grad, v, method='L-BFGS-B', jac=True,
                          options={'maxiter': 50, 'ftol': tol, 'gtol': tol})
            v = res.x
        except Exception:
            break
        
        # Check if constraints are satisfied
        x = v[0:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        W = v[3*N]
        H = 2.0 - W
        
        max_viol = 0
        for i in range(N):
            max_viol = max(max_viol, max(0, r[i] - x[i]), max(0, x[i] + r[i] - W))
            max_viol = max(max_viol, max(0, r[i] - y[i]), max(0, y[i] + r[i] - H))
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                max_viol = max(max_viol, max(0, r[i] + r[j] - dist))
        
        if max_viol < 1e-8:
            break
        
        # Increase penalty
        mu = min(mu * mu_factor, max_mu)
    
    return v

def _polish(x, y, r, W, H, max_iter=50, tol=1e-12):
    """High-precision polish: optimize radii with fixed positions, then adjust positions."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for _ in range(max_iter):
        # First: optimize radii given positions
        # For each circle, max radius is limited by:
        # 1. Distance to walls
        # 2. Distance to other circles
        
        # Compute max possible radius for each circle
        r_max = np.zeros(N)
        for i in range(N):
            # Wall constraints
            r_max[i] = min(x[i], W - x[i], y[i], H - y[i])
            
            # Circle constraints
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                # r_i <= dist - r_j
                r_max[i] = min(r_max[i], dist - r[j])
        
        r_max = np.maximum(r_max, 0)
        
        # Set radii to max possible (with small margin)
        r_new = np.minimum(r, r_max * 0.999)
        r_new = np.maximum(r_new, 1e-6)
        
        if np.allclose(r, r_new, atol=tol):
            break
        r = r_new
    
    # Then: try to improve positions
    # Use SLSQP with constraints
    def objective(v):
        return -np.sum(v[2*N:3*N])
    
    # Constraints: no overlap, within bounds
    cons = []
    for i in range(N):
        for j in range(i+1, N):
            cons.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: np.sqrt((v[i]-v[N+j])**2 + (v[N+i]-v[2*N+j])**2) - (v[2*N+i] + v[2*N+j])
            })
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*N+i]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[N+i] - v[2*N+i]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] + v[2*N+i] - W})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[N+i] + v[2*N+i] - H})
    
    v0 = np.concatenate([x, y, r])
    bounds = [(0, W)]*N + [(0, H)]*N + [(1e-6, min(W, H)/2)]*N
    
    try:
        res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons,
                      options={'maxiter': 30, 'ftol': tol})
        if res.success:
            x_new = res.x[0:N]
            y_new = res.x[N:2*N]
            r_new = res.x[2*N:3*N]
            if _check_validity(x_new, y_new, r_new, W, H, tol=1e-10):
                x, y, r = x_new, y_new, r_new
    except Exception:
        pass
    
    return x, y, r

def _grid_initialization(W, H, rows=5, cols=5):
    """Create initial grid layout."""
    x = []
    y = []
    r = []
    
    cell_w = W / cols
    cell_h = H / rows
    
    count = 0
    for i in range(rows):
        for j in range(cols):
            if count >= N:
                break
            cx = (j + 0.5) * cell_w
            cy = (i + 0.5) * cell_h
            r_init = min(cell_w, cell_h) / 2 * 0.95
            x.append(cx)
            y.append(cy)
            r.append(r_init)
            count += 1
    
    return np.array(x), np.array(y), np.array(r)

def _hexagonal_initialization(W, H):
    """Create hexagonal packing initial layout."""
    x = []
    y = []
    r = []
    
    # Estimate radius for hexagonal packing
    # Area per circle in hexagonal packing: 2*sqrt(3)*r^2
    # Total area: W*H
    # N circles: N * 2*sqrt(3)*r^2 <= W*H
    r_est = np.sqrt(W*H / (N * 2*np.sqrt(3)))
    r_est *= 0.9  # margin
    
    row_height = r_est * np.sqrt(3)
    col_width = 2 * r_est
    
    count = 0
    row = 0
    while count < N:
        offset = (row % 2) * col_width / 2
        col = 0
        while count < N:
            cx = offset + col * col_width
            cy = row * row_height
            if cx + r_est > W or cy + r_est > H:
                break
            x.append(cx)
            y.append(cy)
            r.append(r_est)
            count += 1
            col += 1
        row += 1
    
    # Trim to N
    x = x[:N]
    y = y[:N]
    r = r[:N]
    
    if len(x) < N:
        # Fill remaining with grid
        x_g, y_g, r_g = _grid_initialization(W, H)
        x = np.concatenate([x, x_g[N-len(x):]])
        y = np.concatenate([y, y_g[N-len(y):]])
        r = np.concatenate([r, r_g[N-len(r):]])
    
    return np.array(x), np.array(y), np.array(r)

def _random_initialization(W, H, seed=42):
    """Create random initial layout."""
    np.random.seed(seed)
    x = np.random.uniform(0.1*W, 0.9*W, N)
    y = np.random.uniform(0.1*H, 0.9*H, N)
    r = np.random.uniform(0.05*min(W, H), 0.15*min(W, H), N)
    return x, y, r

def _best_for_aspect(W, H):
    """Try multiple initializations and return best result."""
    best_x, best_y, best_r = None, None, None
    best_sum_r = -np.inf
    
    initializations = [
        _grid_initialization(W, H),
        _hexagonal_initialization(W, H),
        _random_initialization(W, H, seed=42),
        _random_initialization(W, H, seed=123),
    ]
    
    for x0, y0, r0 in initializations:
        # Separate and fix
        x, y, r = _separate_circles(x0, y0, r0, W, H)
        
        # Local optimize
        v = np.concatenate([x, y, r, [W]])
        v_opt = _local_optimize(v, max_iter=50, mu_start=10.0)
        
        x_opt = v_opt[0:N]
        y_opt = v_opt[N:2*N]
        r_opt = v_opt[2*N:3*N]
        W_opt = v_opt[3*N]
        H_opt = 2.0 - W_opt
        
        # Polish
        x_pol, y_pol, r_pol = _polish(x_opt, y_opt, r_opt, W_opt, H_opt)
        
        sum_r = np.sum(r_pol)
        if sum_r > best_sum_r:
            best_sum_r = sum_r
            best_x, best_y, best_r = x_pol, y_pol, r_pol
    
    return best_x, best_y, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s limit
    
    # Try different aspect ratios
    best_result = None
    best_value = -np.inf
    
    aspect_ratios = [
        (1.0, 1.0),      # square
        (1.2, 0.8),      # wider
        (0.8, 1.2),      # taller
        (1.5, 0.5),      # very wide
        (0.5, 1.5),      # very tall
        (1.3, 0.7),
        (0.7, 1.3),
    ]
    
    for W, H in aspect_ratios:
        if time.time() > deadline:
            break
        
        try:
            x, y, r = _best_for_aspect(W, H)
            
            if x is None:
                continue
            
            # Validate
            if not _check_validity(x, y, r, W, H):
                # Try to fix
                x, y, r = _separate_circles(x, y, r, W, H)
                if not _check_validity(x, y, r, W, H):
                    continue
            
            value = np.sum(r)
            if value > best_value:
                best_value = value
                best_result = np.column_stack((x, y, r))
                
        except Exception as e:
            continue
    
    # If no valid result, fallback to simple grid
    if best_result is None:
        W, H = 1.0, 1.0
        x, y, r = _grid_initialization(W, H, rows=5, cols=5)
        x, y, r = _separate_circles(x, y, r, W, H)
        best_result = np.column_stack((x, y, r))
    
    return best_result
