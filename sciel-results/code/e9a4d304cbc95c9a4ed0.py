import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 21

def _objective(z, w, h):
    """
    z: array of shape (3*N,) containing [x1, y1, r1, ..., xN, yN, rN]
    Objective: -sum(r) + penalty for violations
    """
    x = z[0::3]
    y = z[1::3]
    r = z[2::3]
    
    obj = -np.sum(r)
    
    # Wall violations: r_i <= x_i, r_i <= w-x_i, r_i <= y_i, r_i <= h-y_i
    # We want r_i + x_i >= 0 (always true if x>=0, r>=0), but r_i <= x_i means x_i - r_i >= 0
    # Penalty: sum of squares of negative margins
    v_x_min = np.maximum(0, r - x)
    v_x_max = np.maximum(0, r - (w - x))
    v_y_min = np.maximum(0, r - y)
    v_y_max = np.maximum(0, r - (h - y))
    
    wall_pen = np.sum(v_x_min**2 + v_x_max**2 + v_y_min**2 + v_y_max**2)
    obj += 100.0 * wall_pen
    
    # Circle overlap violations: d_ij >= r_i + r_j
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d_sq = dx*dx + dy*dy
            d = np.sqrt(d_sq + 1e-12)
            
            r_sum = r[i] + r[j]
            if d < r_sum:
                v = r_sum - d
                obj += 100.0 * v*v
                
    return obj

def _grad(z, w, h):
    """
    Gradient of _objective
    """
    x = z[0::3]
    y = z[1::3]
    r = z[2::3]
    
    grad = np.zeros_like(z)
    grad[2::3] = -1.0
    
    # Wall gradients
    v_x_min = np.maximum(0, r - x)
    v_x_max = np.maximum(0, r - (w - x))
    v_y_min = np.maximum(0, r - y)
    v_y_max = np.maximum(0, r - (h - y))
    
    # d/dx_i (r_i - x_i)^2 = -2(r_i - x_i) if r_i > x_i
    grad[0::3] -= 200.0 * v_x_min
    # d/dx_i (r_i - (w-x_i))^2 = d/dx_i (r_i - w + x_i)^2 = 2(r_i - w + x_i)
    grad[0::3] += 200.0 * v_x_max
    # d/dy_i (r_i - y_i)^2 = -2(r_i - y_i)
    grad[1::3] -= 200.0 * v_y_min
    # d/dy_i (r_i - (h-y_i))^2 = 2(r_i - h + y_i)
    grad[1::3] += 200.0 * v_y_max
    # d/dr_i terms
    grad[2::3] += 200.0 * (v_x_min + v_x_max + v_y_min + v_y_max)
    
    # Circle overlap gradients
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d_sq = dx*dx + dy*dy
            d = np.sqrt(d_sq + 1e-12)
            
            r_sum = r[i] + r[j]
            if d < r_sum:
                v = r_sum - d
                # Penalty term: 100 * v^2 = 100 * (r_i + r_j - d)^2
                # d/dx_i (r_i + r_j - d)^2 = 2(r_i + r_j - d) * (-dx/d)
                # d/dx_j (r_i + r_j - d)^2 = 2(r_i + r_j - d) * (dx/d)
                # d/dr_i = 2v, d/dr_j = 2v
                
                coeff = 200.0 * v / d
                
                grad[3*i + 0] -= coeff * dx
                grad[3*j + 0] += coeff * dx
                grad[3*i + 1] -= coeff * dy
                grad[3*j + 1] += coeff * dy
                grad[3*i + 2] += 200.0 * v
                grad[3*j + 2] += 200.0 * v
                
    return grad

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

def _hexagonal_start(w, h):
    """
    Generate a hexagonal packing initial guess.
    """
    # Estimate grid size
    # For hexagonal packing, rows are offset.
    # Let's try to fit N circles.
    # Approximate radius r.
    # Area of rectangle = w*h.
    # Area of N circles ~ N * pi * r^2.
    # Packing density ~ 0.9 for hex.
    # N * pi * r^2 * 0.9 <= w * h
    # r ~ sqrt(w * h / (N * pi * 0.9))
    
    r_est = np.sqrt(w * h / (N * np.pi * 0.9))
    
    # Grid dimensions
    # Hex spacing: dx = 2*r, dy = sqrt(3)*r
    # Let's just use a rectangular grid for simplicity in init, then optimize.
    # But hexagonal is better for init.
    
    # Let's try to determine cols and rows.
    # If we use a rectangular grid:
    # cols = ceil(sqrt(N * w/h))
    cols = int(np.ceil(np.sqrt(N * w / h)))
    rows = int(np.ceil(N / cols))
    
    z0 = np.zeros(3 * N)
    for i in range(N):
        col = i % cols
        row = i // cols
        
        # Hexagonal offset
        x_base = (col + 0.5) * (2 * r_est)
        if row % 2 == 1:
            x_base += r_est
            
        y_base = (row + 0.5) * (np.sqrt(3) * r_est)
        
        # Scale to fit in w, h
        # We need to map the generated hex grid to [0, w] x [0, h]
        # But the aspect ratio of the hex grid might not match w/h.
        # So we just use a simple grid scaled to w, h.
        
        x = (col + 0.5) * w / cols
        y = (row + 0.5) * h / rows
        
        # Add hex offset if possible?
        # For now, simple grid is safer for init to avoid immediate overlaps.
        # We can add jitter.
        
        z0[3*i + 0] = x
        z0[3*i + 1] = y
        z0[3*i + 2] = min(w / cols, h / rows) * 0.45 # 90% of half-cell / 2?
        # Cell width = w/cols, height = h/rows.
        # Max radius in cell = min(w/cols, h/rows)/2.
        # Let's start smaller to allow movement.
        z0[3*i + 2] = min(w / cols, h / rows) * 0.4

    return z0

def _random_start(w, h):
    """
    Generate a random initial guess.
    """
    z0 = np.zeros(3 * N)
    # Place in a grid to avoid initial overlaps
    cols = int(np.ceil(np.sqrt(N * w / h)))
    rows = int(np.ceil(N / cols))
    
    for i in range(N):
        col = i % cols
        row = i // cols
        z0[3*i + 0] = (col + 0.5) * w / cols + np.random.uniform(-0.1, 0.1) * (w/cols)
        z0[3*i + 1] = (row + 0.5) * h / rows + np.random.uniform(-0.1, 0.1) * (h/rows)
        z0[3*i + 2] = min(w / cols, h / rows) * 0.4
        
    # Clip
    z0[0::3] = np.clip(z0[0::3], 0, w)
    z0[1::3] = np.clip(z0[1::3], 0, h)
    z0[2::3] = np.maximum(0, z0[2::3])
    
    return z0

def _pack_for_aspect(w, h, deadline, num_trials=5):
    """
    Pack circles in w x h rectangle.
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
        if time.time() > deadline - 5:
            break
            
        if trial == 0:
            z0 = _hexagonal_start(w, h)
        else:
            z0 = _random_start(w, h)
            
        try:
            res = minimize(_objective, z0, jac=_grad, args=(w, h), method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
            
            if res.fun < 0: # Heuristic: if we improved from initial (which is usually negative due to -sum(r))
                z_opt = res.x
                pos = np.column_stack((z_opt[0::3], z_opt[1::3]))
                radii = z_opt[2::3]
                
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
    deadline = start_time + 140 # 10s safety margin
    
    np.random.seed(42)
    
    best_global_score = 0
    best_global_result = None
    
    # Try multiple aspect ratios
    # Perimeter 4 => 2(w+h)=4 => w+h=2
    # Aspect ratio r = w/h. w = 2r/(1+r), h = 2/(1+r)
    # Optimal aspect ratio for circle packing is often close to 1 or sqrt(3)/2 etc.
    # For N=21, a 3x7 or similar rectangular arrangement might be good.
    # 3x7 grid: aspect ratio 7/3 = 2.33? Or 3/7?
    # Hexagonal cluster of 21:
    # Row 1: 3
    # Row 2: 4
    # Row 3: 5
    # Row 4: 4
    # Row 5: 3
    # Total: 3+4+5+4+3 = 19. Not 21.
    # Row 1: 4
    # Row 2: 5
    # Row 3: 5
    # Row 4: 4
    # Row 5: 3
    # Total: 21.
    # Width approx 5 diameters, Height approx 4.5 heights.
    # w/h ~ 5 / 4.5 * (1/sqrt(3))?
    # Let's just try a range.
    
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.66, 2.0, 0.5, 1.33, 0.75]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break
        
        w = 2 * ar / (1 + ar)
        h = 2 / (1 + ar)
        
        pos, radii, score = _pack_for_aspect(w, h, deadline, num_trials=3)
        
        if pos is None:
            continue
        
        if score > best_global_score:
            best_global_score = score
            best_global_result = np.column_stack([pos, radii])
    
    # Fallback if no valid result
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
