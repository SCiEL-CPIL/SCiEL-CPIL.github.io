"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
TIME_LIMIT = 115.0  # safety margin before 120s kill

def _validate(arr):
    """Check if array of (x, y, r) is valid: disjoint, inside rect with perimeter <= 4."""
    if arr.shape != (N, 3):
        return False
    if np.any(arr[:, 2] <= 0):
        return False
    # Bounding box
    xs = arr[:, 0]
    ys = arr[:, 1]
    rs = arr[:, 2]
    left = np.min(xs - rs)
    right = np.max(xs + rs)
    top = np.max(ys + rs)
    bottom = np.min(ys - rs)
    W = right - left
    H = top - bottom
    if W <= 0 or H <= 0:
        return False
    if 2 * (W + H) > 4.0 + 1e-7:
        return False
    # Check disjointness
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - 1e-7:
                return False
    return True

def _score(arr):
    """Compute the raw value: sum of radii, if valid."""
    if not _validate(arr):
        return 0.0
    return float(np.sum(arr[:, 2]))

def _separate_circles(x, y, r, W, H, n_iter=100, tol=1e-10):
    """Separate overlapping circles using iterative projection."""
    for _ in range(n_iter):
        moved = False
        for i in range(N):
            # Keep in bounds
            x[i] = max(r[i], min(W - r[i], x[i]))
            y[i] = max(r[i], min(H - r[i], y[i]))
        
        # Separate pairs
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist - tol:
                    if dist < 1e-12:
                        # Random push
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle)
                        dy = np.sin(angle)
                        dist = 1.0
                    push = (min_dist - dist) / 2.0
                    ux = dx / dist
                    uy = dy / dist
                    x[i] -= push * ux
                    y[i] -= push * uy
                    x[j] += push * ux
                    y[j] += push * uy
                    moved = True
        
        if not moved:
            break
    return x, y

def _force_directed_layout(W, H, initial_x, initial_y, r_init, n_iter=200):
    """Use force-directed layout to find a good configuration."""
    x = initial_x.copy()
    y = initial_y.copy()
    r = r_init.copy()
    
    # Adaptive forces
    for iteration in range(n_iter):
        if time.time() - t0_global > TIME_LIMIT - 5:
            break
        
        # Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Repulsion between circles
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    force = (min_dist - dist) * 0.5
                    if dist > 1e-10:
                        fx[i] += force * dx / dist
                        fy[i] += force * dy / dist
                        fx[j] -= force * dx / dist
                        fy[j] -= force * dy / dist
                    else:
                        angle = np.random.uniform(0, 2 * np.pi)
                        fx[i] += force * np.cos(angle)
                        fy[i] += force * np.sin(angle)
                        fx[j] -= force * np.cos(angle)
                        fy[j] -= force * np.sin(angle)
        
        # Attraction to center (weak)
        for i in range(N):
            fx[i] += (W / 2 - x[i]) * 0.01
            fy[i] += (H / 2 - y[i]) * 0.01
        
        # Apply forces with damping
        damping = 0.95
        x += fx * damping
        y += fy * damping
        
        # Keep in bounds
        for i in range(N):
            x[i] = max(r[i], min(W - r[i], x[i]))
            y[i] = max(r[i], min(H - r[i], y[i]))
    
    return x, y

def _objective_and_grad(v, W, H):
    """Objective: minimize -sum(r) + penalty for violations."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:3*N]
    
    # Base objective
    obj = -np.sum(r)
    grad = np.zeros(3*N)
    grad[2*N:3*N] = -1.0
    
    # Penalty for going out of bounds
    penalty_weight = 100.0
    for i in range(N):
        # Left wall
        viol = r[i] - x[i]
        if viol > 0:
            obj += penalty_weight * viol**2
            grad[i] += 2 * penalty_weight * viol
            grad[2*N+i] += 2 * penalty_weight * viol
        # Right wall
        viol = x[i] + r[i] - W
        if viol > 0:
            obj += penalty_weight * viol**2
            grad[i] += 2 * penalty_weight * viol
            grad[2*N+i] += 2 * penalty_weight * viol
        # Bottom wall
        viol = r[i] - y[i]
        if viol > 0:
            obj += penalty_weight * viol**2
            grad[N+i] += 2 * penalty_weight * viol
            grad[2*N+i] += 2 * penalty_weight * viol
        # Top wall
        viol = y[i] + r[i] - H
        if viol > 0:
            obj += penalty_weight * viol**2
            grad[N+i] += 2 * penalty_weight * viol
            grad[2*N+i] += 2 * penalty_weight * viol
    
    # Penalty for overlapping circles
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r[i] + r[j]
            viol = min_dist - dist
            if viol > 0:
                obj += penalty_weight * viol**2
                if dist > 1e-10:
                    ux = dx / dist
                    uy = dy / dist
                    grad[i] += 2 * penalty_weight * viol * ux
                    grad[N+i] += 2 * penalty_weight * viol * uy
                    grad[j] -= 2 * penalty_weight * viol * ux
                    grad[N+j] -= 2 * penalty_weight * viol * uy
                    grad[2*N+i] += 2 * penalty_weight * viol
                    grad[2*N+j] += 2 * penalty_weight * viol
                else:
                    angle = np.random.uniform(0, 2*np.pi)
                    grad[i] += 2 * penalty_weight * viol * np.cos(angle)
                    grad[N+i] += 2 * penalty_weight * viol * np.sin(angle)
                    grad[j] -= 2 * penalty_weight * viol * np.cos(angle)
                    grad[N+j] -= 2 * penalty_weight * viol * np.sin(angle)
                    grad[2*N+i] += 2 * penalty_weight * viol
                    grad[2*N+j] += 2 * penalty_weight * viol
    
    return obj, grad

def _local_optimize(x, y, r, W, H, max_iter=50):
    """Local optimization using SLSQP-like approach with penalty method."""
    v = np.concatenate([x, y, r])
    
    best_obj = np.inf
    best_v = v.copy()
    
    for iteration in range(max_iter):
        if time.time() - t0_global > TIME_LIMIT - 5:
            break
        
        obj, grad = _objective_and_grad(v, W, H)
        
        if obj < best_obj:
            best_obj = obj
            best_v = v.copy()
        
        # Simple gradient descent with adaptive step
        step_size = 0.01
        for _ in range(10):
            v_new = v - step_size * grad
            # Clip to reasonable bounds
            v_new[:N] = np.clip(v_new[:N], 0, W)
            v_new[N:2*N] = np.clip(v_new[N:2*N], 0, H)
            v_new[2*N:3*N] = np.clip(v_new[2*N:3*N], 0, min(W, H))
            
            obj_new, _ = _objective_and_grad(v_new, W, H)
            if obj_new < obj:
                v = v_new
                obj = obj_new
                break
            step_size *= 0.5
        
        # Check convergence
        if np.max(np.abs(grad)) < 1e-8:
            break
    
    return best_v

def _hexagonal_initialization(W, H):
    """Create hexagonal packing initialization."""
    x = []
    y = []
    
    # Hexagonal lattice
    row_height = 0.866  # sqrt(3)/2
    col_width = 1.0
    
    # Scale to fit
    scale = min(W / (4 * col_width), H / (4 * row_height))
    
    for row in range(5):
        offset = (row % 2) * 0.5 * col_width * scale
        for col in range(5):
            cx = (col + 0.5) * col_width * scale + offset
            cy = (row + 0.5) * row_height * scale
            if cx < W and cy < H:
                x.append(cx)
                y.append(cy)
    
    # Take first 21
    x = np.array(x[:N])
    y = np.array(y[:N])
    
    # Compute max radius
    min_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            min_dist = min(min_dist, d)
    for i in range(N):
        min_dist = min(min_dist, x[i], W - x[i], y[i], H - y[i])
    
    r = np.full(N, min_dist / 2 * 0.95)  # Slight margin
    
    return x, y, r

def _grid_initialization(W, H, cols=5, rows=5, remove_corners=True):
    """Create grid initialization."""
    positions = []
    for i in range(cols):
        for j in range(rows):
            if remove_corners and ((i, j) in [(0,0), (0,rows-1), (cols-1,0), (cols-1,rows-1)]):
                continue
            positions.append((i, j))
    
    positions = positions[:N]
    x = np.array([p[0] / (cols - 1) * W for p in positions])
    y = np.array([p[1] / (rows - 1) * H for p in positions])
    
    # Compute max radius
    min_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            min_dist = min(min_dist, d)
    for i in range(N):
        min_dist = min(min_dist, x[i], W - x[i], y[i], H - y[i])
    
    r = np.full(N, min_dist / 2 * 0.95)
    
    return x, y, r

def _random_initialization(W, H):
    """Create random initialization."""
    x = np.random.uniform(0.1, W - 0.1, N)
    y = np.random.uniform(0.1, H - 0.1, N)
    
    # Start with small radii
    r = np.full(N, 0.05)
    
    # Separate
    x, y = _separate_circles(x, y, r, W, H, n_iter=50)
    
    # Try to grow radii
    for _ in range(20):
        min_dist = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                min_dist = min(min_dist, d)
        for i in range(N):
            min_dist = min(min_dist, x[i], W - x[i], y[i], H - y[i])
        
        new_r = min_dist / 2 * 0.95
        if new_r > np.max(r) * 1.01:
            r = np.full(N, new_r)
        else:
            break
    
    return x, y, r

def _best_for_aspect(W, H, init_method='grid'):
    """Find best configuration for given aspect ratio."""
    if init_method == 'hex':
        x, y, r = _hexagonal_initialization(W, H)
    elif init_method == 'random':
        x, y, r = _random_initialization(W, H)
    else:
        x, y, r = _grid_initialization(W, H)
    
    # Force-directed refinement
    x, y = _force_directed_layout(W, H, x, y, r, n_iter=100)
    
    # Recompute optimal radii
    min_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            min_dist = min(min_dist, d)
    for i in range(N):
        min_dist = min(min_dist, x[i], W - x[i], y[i], H - y[i])
    
    r = np.full(N, min_dist / 2)
    
    # Local optimization
    v = _local_optimize(x, y, r, W, H, max_iter=30)
    x_opt = v[:N]
    y_opt = v[N:2*N]
    r_opt = v[2*N:3*N]
    
    # Ensure validity
    x_opt = np.clip(x_opt, r_opt, W - r_opt)
    y_opt = np.clip(y_opt, r_opt, H - r_opt)
    
    arr = np.column_stack([x_opt, y_opt, r_opt])
    return arr

t0_global = time.time()

def construct():
    global t0_global
    t0_global = time.time()
    
    best_arr = None
    best_score = 0.0
    
    # Try multiple aspect ratios
    aspect_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    
    for ar in aspect_ratios:
        if time.time() - t0_global > TIME_LIMIT - 10:
            break
        
        W = 2 * ar / (ar + 1)
        H = 2 / (ar + 1)
        
        # Try different initializations
        for init_method in ['grid', 'hex', 'random']:
            if time.time() - t0_global > TIME_LIMIT - 10:
                break
            
            arr = _best_for_aspect(W, H, init_method)
            s = _score(arr)
            
            if s > best_score:
                best_score = s
                best_arr = arr.copy()
    
    # Final polish: try to improve the best solution
    if best_arr is not None and time.time() - t0_global < TIME_LIMIT - 5:
        # Extract W, H from bounding box
        xs = best_arr[:, 0]
        ys = best_arr[:, 1]
        rs = best_arr[:, 2]
        left = np.min(xs - rs)
        right = np.max(xs + rs)
        top = np.max(ys + rs)
        bottom = np.min(ys - rs)
        W = right - left
        H = top - bottom
        
        # Normalize to [0, W] x [0, H]
        x = xs - left
        y = ys - bottom
        r = rs
        
        # Final local optimization
        v = _local_optimize(x, y, r, W, H, max_iter=50)
        x_opt = v[:N]
        y_opt = v[N:2*N]
        r_opt = v[2*N:3*N]
        
        x_opt = np.clip(x_opt, r_opt, W - r_opt)
        y_opt = np.clip(y_opt, r_opt, H - r_opt)
        
        arr = np.column_stack([x_opt, y_opt, r_opt])
        s = _score(arr)
        
        if s > best_score:
            best_score = s
            best_arr = arr.copy()
    
    if best_arr is None:
        # Fallback: simple grid
        W = 1.0
        H = 1.0
        x = np.linspace(0.1, 0.9, N)
        y = np.linspace(0.1, 0.9, N)
        r = np.full(N, 0.04)
        best_arr = np.column_stack([x, y, r])
    
    return best_arr
