import numpy as np
from scipy.optimize import minimize
import time
import random
import math

def _rect_from_width(w):
    """Given width w, compute height h such that 2*(w+h)=4, i.e., h = 2 - w."""
    return 2.0 - w

def _evaluate(circles, w):
    """
    Evaluate the sum of radii given circle positions (x, y, r) and rectangle width w.
    Returns (score, is_valid, penalty).
    """
    h = _rect_from_width(w)
    if w <= 0 or h <= 0:
        return 0.0, False, 1e6
    
    n = len(circles)
    # Check bounds
    penalty = 0.0
    for i in range(n):
        x, y, r = circles[i]
        # Circle must be inside rectangle
        if x < r or x > w - r or y < r or y > h - r or r < 0:
            penalty += max(0, r - x) + max(0, x - (w - r)) + max(0, r - y) + max(0, y - (h - r)) + max(0, -r)
    
    # Check non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            if dist < ri + rj:
                penalty += (ri + rj - dist)
    
    total_r = sum(c[2] for c in circles)
    is_valid = penalty < 1e-9
    return total_r, is_valid, penalty

def _grid_layout(w, n_rows, n_cols):
    """Create a grid layout of circles."""
    h = _rect_from_width(w)
    circles = []
    # Use a simple grid for initialization
    r_est = min(w / n_cols, h / n_rows) * 0.45
    for i in range(n_rows):
        for j in range(n_cols):
            if len(circles) >= 21:
                break
            x = (j + 0.5) * w / n_cols
            y = (i + 0.5) * h / n_rows
            circles.append([x, y, r_est])
    
    # If we have fewer than 21, add more
    while len(circles) < 21:
        x = random.uniform(0.2 * w, 0.8 * w)
        y = random.uniform(0.2 * h, 0.8 * h)
        r_est = min(w, h) * 0.1
        circles.append([x, y, r_est])
    
    return circles[:21]

def _hexagonal_layout(w, n_rows):
    """Create a hexagonal layout."""
    h = _rect_from_width(w)
    circles = []
    
    # Determine number of columns per row
    # For hexagonal packing, alternate between ceil and floor of n/rows
    n_cols_alt = []
    for i in range(n_rows):
        if 21 % n_rows == 0:
            n_cols_alt.append(21 // n_rows)
        else:
            # Distribute remainder
            if i < 21 % n_rows:
                n_cols_alt.append(21 // n_rows + 1)
            else:
                n_cols_alt.append(21 // n_rows)
    
    total_cols = sum(n_cols_alt)
    
    # Estimate radius
    # For hexagonal packing: horizontal spacing = 2r, vertical spacing = r*sqrt(3)
    # Total width ~ max_cols * 2r
    # Total height ~ n_rows * r*sqrt(3) + 2r
    max_cols = max(n_cols_alt)
    r_est = min(w / (max_cols * 2.0), h / (n_rows * math.sqrt(3) + 2.0))
    
    r = r_est * 0.95
    
    y_offset = r
    for i in range(n_rows):
        n_cols = n_cols_alt[i]
        # Offset odd rows
        if i % 2 == 1:
            x_start = r + r  # offset by one radius
            # Adjust to center
            total_width = (n_cols - 1) * 2 * r
            x_start = (w - total_width) / 2.0
        else:
            total_width = (n_cols - 1) * 2 * r
            x_start = (w - total_width) / 2.0
        
        y = r + i * r * math.sqrt(3)
        
        for j in range(n_cols):
            if len(circles) >= 21:
                break
            x = x_start + j * 2 * r
            circles.append([x, y, r])
    
    # Fill remaining
    while len(circles) < 21:
        x = random.uniform(r, w - r)
        y = random.uniform(r, h - r)
        circles.append([x, y, r * 0.8])
    
    return circles[:21]

def _random_layout(w, n_circles=21):
    """Create a random layout."""
    h = _rect_from_width(w)
    circles = []
    r_est = min(w, h) / 6.0
    
    for _ in range(n_circles):
        x = random.uniform(r_est, w - r_est)
        y = random.uniform(r_est, h - r_est)
        r = r_est * random.uniform(0.5, 1.0)
        circles.append([x, y, r])
    
    return circles

def _objective_and_gradient(params, w):
    """
    Objective function with analytic gradient.
    params: flat array of length 3*n where n=21
    Returns (objective, gradient)
    """
    n = 21
    h = _rect_from_width(w)
    circles = params.reshape((n, 3))
    
    # Objective: maximize sum of radii, so minimize -sum(r) + penalty
    obj = -sum(circles[:, 2])
    grad = np.zeros_like(params)
    grad[2::3] = -1.0  # d(-sum(r))/d(r_i) = -1
    
    # Penalty for bounds
    for i in range(n):
        x, y, r = circles[i]
        # x bounds: x >= r, x <= w - r
        if x < r:
            p = r - x
            obj += p ** 2
            grad[3*i] += -2*p       # dx
            grad[3*i+2] += 2*p       # dr
        if x > w - r:
            p = x - (w - r)
            obj += p ** 2
            grad[3*i] += 2*p         # dx
            grad[3*i+2] += -2*p      # dr
        # y bounds
        if y < r:
            p = r - y
            obj += p ** 2
            grad[3*i+1] += -2*p      # dy
            grad[3*i+2] += 2*p       # dr
        if y > h - r:
            p = y - (h - r)
            obj += p ** 2
            grad[3*i+1] += 2*p       # dy
            grad[3*i+2] += -2*p      # dr
        # r >= 0
        if r < 0:
            p = -r
            obj += p ** 2
            grad[3*i+2] += -2*p      # dr
    
    # Penalty for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx ** 2 + dy ** 2
            dist = math.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-6
            sum_r = ri + rj
            if dist < sum_r:
                p = sum_r - dist
                obj += p ** 2
                # d(p^2)/d(x_i) = 2p * dp/d(x_i)
                # dp/d(x_i) = -dx/dist (since p = sum_r - dist, d(dist)/d(x_i) = dx/dist)
                inv_dist = 1.0 / dist
                grad[3*j] += -2*p * dx * inv_dist   # d(x_j) term
                grad[3*j+1] += -2*p * dy * inv_dist # d(y_j) term
                grad[3*i] += 2*p * dx * inv_dist    # d(x_i) term
                grad[3*i+1] += 2*p * dy * inv_dist  # d(y_i) term
                grad[3*i+2] += 2*p                  # d(r_i)
                grad[3*j+2] += 2*p                  # d(r_j)
    
    return obj, grad

def _optimize_from(initial_circles, w, time_limit):
    """
    Optimize from an initial layout using SLSQP with analytic gradient.
    """
    n = 21
    h = _rect_from_width(w)
    
    # Set initial bounds
    bounds = []
    for i in range(n):
        # x: [0, w]
        bounds.append((0, w))
        # y: [0, h]
        bounds.append((0, h))
        # r: [0, min(w, h)/2]
        bounds.append((0, min(w, h) / 2))
    
    x0 = np.array([c[0] for c in initial_circles] + 
                   [c[1] for c in initial_circles] + 
                   [c[2] for c in initial_circles])
    # Reorder to (x0, y0, r0, x1, y1, r1, ...)
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = initial_circles[i][0]
        x0[3*i+1] = initial_circles[i][1]
        x0[3*i+2] = initial_circles[i][2]
    
    # Use SLSQP
    start = time.time()
    
    def fun_and_grad(params):
        return _objective_and_gradient(params, w)
    
    try:
        res = minimize(fun_and_grad, x0, jac=True, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': 200, 'ftol': 1e-12, 'disp': False})
    except Exception as e:
        return initial_circles, 0.0
    
    if res.success or res.fun < _objective_and_gradient(x0, w)[0]:
        best_params = res.x
    else:
        best_params = x0
    
    # Convert back to circles
    circles = []
    for i in range(n):
        circles.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])
    
    score, valid, pen = _evaluate(circles, w)
    return circles, score

def construct():
    N = 21
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_score = 0.0
    best_circles = None
    best_w = 1.0
    
    # Multi-start over different aspect ratios and layouts
    aspect_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]
    
    # Try hexagonal layouts with different numbers of rows
    for w_ratio in aspect_ratios:
        if time.time() > deadline:
            break
        w = 2.0 * w_ratio / (1.0 + w_ratio)
        if w <= 0 or w >= 2.0:
            continue
        
        h = _rect_from_width(w)
        
        # Try different hexagonal row counts
        for n_rows in [3, 4, 5, 6, 7]:
            if time.time() > deadline:
                break
            try:
                initial = _hexagonal_layout(w, n_rows)
                circles, score = _optimize_from(initial, w, deadline - time.time())
                
                if score > best_score:
                    best_score = score
                    best_circles = circles
                    best_w = w
            except Exception:
                pass
        
        # Try grid layout
        if time.time() > deadline:
            break
        for n_rows in [3, 4, 5, 6]:
            if time.time() > deadline:
                break
            n_cols = max(1, 21 // n_rows)
            try:
                initial = _grid_layout(w, n_rows, n_cols)
                circles, score = _optimize_from(initial, w, deadline - time.time())
                
                if score > best_score:
                    best_score = score
                    best_circles = circles
                    best_w = w
            except Exception:
                pass
        
        # Try random layouts
        if time.time() > deadline:
            break
        for _ in range(3):
            if time.time() > deadline:
                break
            try:
                initial = _random_layout(w)
                circles, score = _optimize_from(initial, w, deadline - time.time())
                
                if score > best_score:
                    best_score = score
                    best_circles = circles
                    best_w = w
            except Exception:
                pass
    
    # If no valid solution found, return a default
    if best_circles is None:
        # Return the original hexagonal packing
        sqrt3 = math.sqrt(3)
        r = 2.0 / (10.0 + 5.0 * sqrt3)
        w = 8.0 * r
        h = 2.0 * r + 5.0 * r * sqrt3
        
        circles = []
        for j in range(4):
            x = r + 2.0 * j * r
            y = r
            circles.append([x, y, r])
        y1 = r + r * sqrt3
        for j in range(3):
            x = 2.0 * r + 2.0 * j * r
            circles.append([x, y1, r])
        y2 = r + 2.0 * r * sqrt3
        for j in range(4):
            x = r + 2.0 * j * r
            circles.append([x, y2, r])
        y3 = r + 3.0 * r * sqrt3
        for j in range(3):
            x = 2.0 * r + 2.0 * j * r
            circles.append([x, y3, r])
        y4 = r + 4.0 * r * sqrt3
        for j in range(4):
            x = r + 2.0 * j * r
            circles.append([x, y4, r])
        y5 = r + 5.0 * r * sqrt3
        for j in range(3):
            x = 2.0 * r + 2.0 * j * r
            circles.append([x, y5, r])
        
        return np.array(circles)
    
    # Verify the best solution is valid
    score, valid, pen = _evaluate(best_circles, best_w)
    if valid:
        return np.array(best_circles)
    
    # If not valid, try to fix or return default
    # Return a scaled-down version
    circles = []
    for c in best_circles:
        x, y, r = c
        r_new = r * 0.95
        x_new = x * 0.95
        y_new = y * 0.95
        circles.append([x_new, y_new, r_new])
    
    score2, valid2, pen2 = _evaluate(circles, best_w)
    if valid2:
        return np.array(circles)
    
    # Final fallback
    sqrt3 = math.sqrt(3)
    r = 2.0 / (10.0 + 5.0 * sqrt3)
    w = 8.0 * r
    h = 2.0 * r + 5.0 * r * sqrt3
    
    circles = []
    for j in range(4):
        x = r + 2.0 * j * r
        y = r
        circles.append([x, y, r])
    y1 = r + r * sqrt3
    for j in range(3):
        x = 2.0 * r + 2.0 * j * r
        circles.append([x, y1, r])
    y2 = r + 2.0 * r * sqrt3
    for j in range(4):
        x = r + 2.0 * j * r
        circles.append([x, y2, r])
    y3 = r + 3.0 * r * sqrt3
    for j in range(3):
        x = 2.0 * r + 2.0 * j * r
        circles.append([x, y3, r])
    y4 = r + 4.0 * r * sqrt3
    for j in range(4):
        x = r + 2.0 * j * r
        circles.append([x, y4, r])
    y5 = r + 5.0 * r * sqrt3
    for j in range(3):
        x = 2.0 * r + 2.0 * j * r
        circles.append([x, y5, r])
    
    return np.array(circles)

