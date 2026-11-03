import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _compute_violations_and_score(pos, radii, w, h):
    """
    Compute constraint violations and the sum of radii.
    Returns (violation_sum, sum_radii, valid_flag)
    """
    n = len(radii)
    # Wall violations
    wall_viol = 0.0
    for i in range(n):
        wall_viol += max(0, radii[i] - pos[i, 0]) ** 2
        wall_viol += max(0, radii[i] - (w - pos[i, 0])) ** 2
        wall_viol += max(0, radii[i] - pos[i, 1]) ** 2
        wall_viol += max(0, radii[i] - (h - pos[i, 1])) ** 2
    
    # Circle-Circle violations
    circle_viol = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            overlap = radii[i] + radii[j] - d
            if overlap > 0:
                circle_viol += overlap ** 2
                
    total_viol = wall_viol + circle_viol
    sum_rad = np.sum(radii)
    # Consider valid if violation is very small
    valid = total_viol < 1e-6
    return total_viol, sum_rad, valid

def _objective(x, w, h, lambda_rad=0.5):
    """
    Objective function for optimization.
    x = [x1, y1, r1, x2, y2, r2, ...]
    We minimize: (Sum of Squared Violations) - lambda_rad * (Sum of Radii)
    This encourages large radii, penalized by overlap/wall breaches.
    """
    n = N
    pos = x[:n*2].reshape(n, 2)
    radii = x[n*2:]
    
    # Compute violations
    wall_viol = 0.0
    for i in range(n):
        wall_viol += max(0, radii[i] - pos[i, 0]) ** 2
        wall_viol += max(0, radii[i] - (w - pos[i, 0])) ** 2
        wall_viol += max(0, radii[i] - pos[i, 1]) ** 2
        wall_viol += max(0, radii[i] - (h - pos[i, 1])) ** 2
        
    circle_viol = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d_sq = dx*dx + dy*dy
            # Optimization: check d_sq > (r_i+r_j)^2 first to avoid sqrt
            r_sum = radii[i] + radii[j]
            if d_sq < r_sum * r_sum:
                d = np.sqrt(d_sq)
                overlap = r_sum - d
                circle_viol += overlap * overlap
                
    total_viol = wall_viol + circle_viol
    sum_rad = np.sum(radii)
    
    return total_viol - lambda_rad * sum_rad

def _greedy_place(w, h, num_circles, grid_res=20):
    """
    Place circles sequentially to maximize the minimum radius at each step.
    Returns pos (N, 2) and radii (N,)
    """
    pos = np.zeros((num_circles, 2))
    radii = np.zeros(num_circles)
    
    # Grid of potential centers
    xs = np.linspace(0, w, grid_res)
    ys = np.linspace(0, h, grid_res)
    
    for k in range(num_circles):
        best_r = 0
        best_x = w / 2.0
        best_y = h / 2.0
        
        for x in xs:
            for y in ys:
                # Max radius based on walls
                r_wall = min(x, w - x, y, h - y)
                if r_wall <= 0:
                    continue
                
                # Max radius based on existing circles
                r_circle = r_wall
                for i in range(k):
                    dx = x - pos[i, 0]
                    dy = y - pos[i, 1]
                    d = np.sqrt(dx*dx + dy*dy)
                    # Distance to surface of circle i
                    dist_to_surface = d - radii[i]
                    if dist_to_surface < r_circle:
                        r_circle = dist_to_surface
                        if r_circle <= 0:
                            break
                
                if r_circle > best_r:
                    best_r = r_circle
                    best_x = x
                    best_y = y
        
        pos[k, 0] = best_x
        pos[k, 1] = best_y
        radii[k] = best_r
        
    return pos, radii

def _pack_for_aspect(w, h, start_time, deadline, num_restarts=3):
    """
    Pack circles in w x h rectangle.
    Returns best (pos, radii) and score found for this aspect.
    """
    best_score = 0
    best_pos = None
    best_radii = None
    
    # Define bounds for L-BFGS-B
    max_r = min(w, h) / 2.0
    bounds = []
    for i in range(N):
        bounds.append((0, w))
        bounds.append((0, h))
        bounds.append((0, max_r))
        
    # 1. Greedy Initialization
    pos_grid, rad_grid = _greedy_place(w, h, N, grid_res=25)
    x0 = np.concatenate([pos_grid.flatten(), rad_grid])
    
    # 2. Random Initialization (optional, maybe skip to save time if greedy is good)
    # Let's try 1 random restart if time permits
    
    initializations = [x0]
    
    if time.time() < deadline - 10:
        pos_rand = np.random.rand(N, 2) * np.array([w, h])
        rad_rand = np.random.rand(N) * 0.1 * min(w, h)
        initializations.append(np.concatenate([pos_rand.flatten(), rad_rand]))

    for x0 in initializations:
        if time.time() > deadline:
            break
            
        try:
            # Use a higher lambda to push radii up, but penalty keeps them valid
            # Since we start valid (greedy), we can afford a higher lambda
            result = minimize(_objective, x0, args=(w, h, 1.0), method='L-BFGS-B', 
                              bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12, 'maxls': 20})
            
            x_opt = result.x
            pos_opt = x_opt[:N*2].reshape(N, 2)
            rad_opt = x_opt[N*2:]
            
            # Verify and clean up
            viol, score, valid = _compute_violations_and_score(pos_opt, rad_opt, w, h)
            
            # If not fully valid, shrink radii slightly
            if not valid:
                if viol > 1e-4:
                    # Try to fix by scaling down
                    scale = 0.95
                    while viol > 1e-6 and scale > 0.1:
                        rad_new = rad_opt * scale
                        viol, s, valid = _compute_violations_and_score(pos_opt, rad_new, w, h)
                        if valid:
                            rad_opt = rad_new
                            score = s
                            break
                        scale *= 0.9
                    if not valid:
                        continue
                
            if score > best_score:
                best_score = score
                best_pos = pos_opt.copy()
                best_radii = rad_opt.copy()
                
        except Exception:
            continue
            
    return best_pos, best_radii, best_score

def construct():
    start_time = time.time()
    # 150s limit. Use 145s for safety.
    deadline = start_time + 145
    
    best_global_score = 0
    best_global_result = None
    
    # Aspect ratios to try
    # Perimeter 4 => 2(w+h)=4 => w+h=2 => h = 2-w
    # w in [0.5, 1.5]
    aspect_ratios = np.linspace(0.5, 1.5, 11)
    
    np.random.seed(42)
    np.random.shuffle(aspect_ratios)
    
    # Add square
    aspect_ratios = np.insert(aspect_ratios, 0, 1.0)
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10: # Leave 10s for cleanup
            break
            
        w = 2 * ar / (1 + ar)
        h = 2 / (1 + ar)
        
        pos, radii, score = _pack_for_aspect(w, h, start_time, deadline, num_restarts=2)
        
        if pos is not None and score > best_global_score:
            best_global_score = score
            # Ensure strict validity
            viol, s, valid = _compute_violations_and_score(pos, radii, w, h)
            if valid:
                best_global_result = np.column_stack([pos, radii])
            elif viol < 1e-3:
                # Fix by shrinking
                scale = 0.99
                while viol > 1e-6 and scale > 0.1:
                    radii_new = radii * scale
                    viol, s, valid = _compute_violations_and_score(pos, radii_new, w, h)
                    if valid:
                        radii = radii_new
                        break
                    scale *= 0.95
                if valid:
                    best_global_result = np.column_stack([pos, radii])
    
    # Fallback if no valid result found
    if best_global_result is None:
        w = 1.0
        h = 1.0
        side = int(np.ceil(np.sqrt(N)))
        pos = np.zeros((N, 2))
        rad = np.zeros(N)
        r_val = 0.5 / side
        for i in range(N):
            col = i % side
            row = i // side
            pos[i, 0] = (col + 0.5) / side
            pos[i, 1] = (row + 0.5) / side
            rad[i] = r_val * 0.9
        best_global_result = np.column_stack([pos, rad])
        
    return best_global_result
