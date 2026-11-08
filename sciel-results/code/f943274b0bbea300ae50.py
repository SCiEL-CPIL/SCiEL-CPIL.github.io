"""Rewritten program for circle packing in perimeter-4 rectangle."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21
PERIMETER = 4.0
TIME_LIMIT = 110.0  # Safety margin before 120s limit


def _compute_value(circles):
    """Compute the sum of radii for a valid configuration."""
    if len(circles) == 0:
        return 0.0
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Check constraints
    if np.any(r <= 0):
        return 0.0
    
    # Check circles are within rectangle [0, W] x [0, H]
    # We need to determine W and H from the bounding box
    x_min = np.min(x - r)
    x_max = np.max(x + r)
    y_min = np.min(y - r)
    y_max = np.max(y + r)
    
    W = x_max - x_min
    H = y_max - y_min
    
    # Check perimeter constraint
    if 2 * (W + H) > PERIMETER + 1e-6:
        return 0.0
    
    # Check all circles are within the bounding rectangle
    # Actually, the bounding rectangle IS defined by the circles, so they're always inside
    # But we need to ensure no circle extends beyond [0,W]x[0,H] which is automatically true
    
    # Check pairwise non-overlap
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0
    
    return np.sum(r)


def _seed_grid():
    """Create a simple grid-based seed configuration."""
    # Try different arrangements
    # For 21 circles, try 5x5 grid (25 cells, use 21)
    # Or 3x7, 7x3, etc.
    
    best_val = -1
    best_circles = None
    
    for cols in range(1, N + 1):
        rows = (N + cols - 1) // cols
        if rows * cols < N:
            continue
            
        # Rectangle dimensions
        # Perimeter = 2(W + H) = 4 => W + H = 2
        # Let's try W = cols * s, H = rows * s where s is cell size
        # But we need 2(W+H) = 4 => W + H = 2
        # So s = 2 / (cols + rows)
        s = 2.0 / (cols + rows)
        W = cols * s
        H = rows * s
        
        circles = []
        for i in range(cols):
            for j in range(rows):
                if len(circles) >= N:
                    break
                cx = (i + 0.5) * s
                cy = (j + 0.5) * s
                r = 0.5 * s - 1e-4
                circles.append([cx, cy, r])
        circles = np.array(circles[:N])
        
        val = _compute_value(circles)
        if val > best_val:
            best_val = val
            best_circles = circles.copy()
    
    return best_circles


def _optimize(circles, time_budget):
    """Optimize circle positions and radii using gradient descent."""
    t_start = time.time()
    circles = circles.copy()
    
    # Variable: x, y, r for each circle
    # We'll use a penalty-based approach
    
    def objective(params):
        if time.time() - t_start > time_budget:
            return 1e10  # Signal timeout
        
        n = len(params) // 3
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        # Penalty for negative radii
        penalty = 0.0
        penalty -= 100.0 * np.sum(np.minimum(r, 0))
        
        # Compute bounding box
        x_min = np.min(x - r)
        x_max = np.max(x + r)
        y_min = np.min(y - r)
        y_max = np.max(y + r)
        
        W = x_max - x_min
        H = y_max - y_min
        
        # Perimeter constraint: 2(W + H) <= 4
        pen_perim = max(0, 2 * (W + H) - PERIMETER)
        penalty += 1000.0 * pen_perim ** 2
        
        # Circle-circle overlap penalty
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    penalty += 1000.0 * overlap ** 2
        
        # Maximize sum of radii => minimize -sum(r)
        return -np.sum(r) + penalty
    
    # Initial parameters
    params = circles.flatten()
    
    # Use Nelder-Mead or BFGS
    result = minimize(objective, params, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    if result.success or result.fun < 1e6:
        new_circles = result.x.reshape(-1, 3)
        # Ensure radii are positive
        new_circles[:, 2] = np.maximum(new_circles[:, 2], 1e-6)
        return new_circles
    
    return circles


def _local_search(circles, time_budget):
    """Perform local search by adjusting radii and positions."""
    t_start = time.time()
    circles = circles.copy()
    n = len(circles)
    
    # Normalize to fit in unit perimeter constraint
    # Compute current bounding box
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    x_min = np.min(x - r)
    x_max = np.max(x + r)
    y_min = np.min(y - r)
    y_max = np.max(y + r)
    
    W = x_max - x_min
    H = y_max - y_min
    perim = 2 * (W + H)
    
    if perim > PERIMETER:
        # Scale down
        scale = PERIMETER / perim
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        circles[:, 0] = center_x + (circles[:, 0] - center_x) * scale
        circles[:, 1] = center_y + (circles[:, 1] - center_y) * scale
        circles[:, 2] = circles[:, 2] * scale
    
    # Try to increase radii while maintaining non-overlap
    for iteration in range(1000):
        if time.time() - t_start > time_budget:
            break
        
        improved = False
        for i in range(n):
            # Try increasing radius of circle i
            r_old = circles[i, 2]
            r_new = r_old * 1.01
            
            # Check if it still fits
            x_new = circles[i, 0]
            y_new = circles[i, 1]
            
            # Check against all other circles
            valid = True
            for j in range(n):
                if i == j:
                    continue
                dx = x_new - circles[j, 0]
                dy = y_new - circles[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r_new + circles[j, 2] - 1e-9:
                    valid = False
                    break
            
            if valid:
                # Check perimeter constraint
                x_min_new = min(np.min(circles[:, 0] - circles[:, 2]), x_new - r_new)
                x_max_new = max(np.max(circles[:, 0] + circles[:, 2]), x_new + r_new)
                y_min_new = min(np.min(circles[:, 1] - circles[:, 2]), y_new - r_new)
                y_max_new = max(np.max(circles[:, 1] + circles[:, 2]), y_new + r_new)
                
                W_new = x_max_new - x_min_new
                H_new = y_max_new - y_min_new
                
                if 2 * (W_new + H_new) <= PERIMETER + 1e-6:
                    circles[i, 2] = r_new
                    improved = True
        
        if not improved:
            break
    
    return circles


def construct():
    """Construct 21 disjoint circles in a perimeter-4 rectangle maximizing sum of radii."""
    t_start = time.time()
    
    # Start with a good seed
    circles = _seed_grid()
    
    if circles is None or len(circles) < N:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        circles = np.array(cells[:N])
    
    # Optimize
    time_remaining = TIME_LIMIT - (time.time() - t_start) - 5.0  # 5s safety margin
    if time_remaining > 2.0:
        circles = _optimize(circles, time_remaining * 0.5)
    
    # Local search
    time_remaining = TIME_LIMIT - (time.time() - t_start) - 3.0
    if time_remaining > 1.0:
        circles = _local_search(circles, time_remaining)
    
    # Final validation and cleanup
    # Ensure exactly N circles
    if len(circles) > N:
        circles = circles[:N]
    elif len(circles) < N:
        # Pad with tiny circles if needed (shouldn't happen)
        while len(circles) < N:
            circles = np.vstack([circles, [[0.5, 0.5, 1e-6]]])
    
    # Ensure radii are positive
    circles[:, 2] = np.maximum(circles[:, 2], 1e-6)
    
    return circles
