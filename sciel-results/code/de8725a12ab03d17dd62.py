"""Rewritten program for circle packing in perimeter-4 rectangle."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21

# EVOLVE-BLOCK-START

def _compute_value(circles):
    """Compute the objective: sum of radii, if valid."""
    if circles.shape[0] != N:
        return -np.inf
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Check radii positive
    if np.any(r <= 0):
        return -np.inf
    
    # Check within [0,1] x [0,1] (perimeter 4 -> unit square)
    if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
        return -np.inf
    if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
        return -np.inf
    
    # Check pairwise non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return -np.inf
    
    return np.sum(r)

def _construct_grid_based():
    """Grid-based cellular decomposition approach."""
    # Use a 5x5 grid, pick 21 cells
    side = 5
    # We'll create a grid of 5x5 = 25 cells, pick 21
    # To maximize sum of radii, we want larger cells
    # Actually, for a uniform grid, each cell has inradius 0.5/side
    # Sum = 21 * 0.5/5 = 2.1
    
    # But we can do better by using non-uniform grids or merging cells
    # Let's try a simple approach: place circles at grid points with max possible radius
    
    # For a 5x5 grid in unit square, spacing = 0.2
    # Max radius per circle = 0.1 (if centered)
    # Sum = 21 * 0.1 = 2.1
    
    # Try 4x5 grid (20 cells) + 1 extra
    # Or use a smarter packing
    
    # Let's start with a simple grid and then optimize
    side = 5
    positions = []
    for i in range(side):
        for j in range(side):
            positions.append(((i + 0.5) / side, (j + 0.5) / side))
    
    # Take first 21 positions
    positions = positions[:N]
    
    # Initial radius estimate
    spacing = 1.0 / side
    r_init = spacing / 2 - 1e-4
    
    circles = np.zeros((N, 3))
    for k, (px, py) in enumerate(positions):
        circles[k] = [px, py, r_init]
    
    return circles

def _optimize_circles(circles_init, deadline):
    """Optimize circle positions and radii using scipy."""
    n = circles_init.shape[0]
    
    def objective(vars_):
        """Minimize negative sum of radii."""
        x = vars_[:n]
        y = vars_[n:2*n]
        r = vars_[2*n:3*n]
        
        # Penalty for constraints
        penalty = 0.0
        eps = 1e-6
        
        # Boundary constraints
        for i in range(n):
            if x[i] - r[i] < 0:
                penalty += (x[i] - r[i]) ** 2
            if x[i] + r[i] > 1:
                penalty += (x[i] + r[i] - 1) ** 2
            if y[i] - r[i] < 0:
                penalty += (y[i] - r[i]) ** 2
            if y[i] + r[i] > 1:
                penalty += (y[i] + r[i] - 1) ** 2
        
        # Overlap constraints
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        return -np.sum(r) + 100 * penalty
    
    x0 = np.concatenate([circles_init[:, 0], circles_init[:, 1], circles_init[:, 2]])
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0, 1))  # x
    for i in range(n):
        bounds.append((0, 1))  # y
    for i in range(n):
        bounds.append((1e-6, 0.5))  # r
    
    # Try multiple random starts
    best_result = None
    best_value = -np.inf
    
    np.random.seed(42)
    n_starts = 5
    
    for start in range(n_starts):
        if time.time() > deadline - 10:
            break
        
        if start == 0:
            x0_trial = x0.copy()
        else:
            # Perturb initial solution
            x0_trial = x0 + np.random.randn(len(x0)) * 0.05
            # Clip to reasonable bounds
            x0_trial[:n] = np.clip(x0_trial[:n], 0.05, 0.95)
            x0_trial[n:2*n] = np.clip(x0_trial[n:2*n], 0.05, 0.95)
            x0_trial[2*n:3*n] = np.clip(x0_trial[2*n:3*n], 0.01, 0.3)
        
        try:
            result = minimize(objective, x0_trial, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        except Exception:
            continue
        
        if result.success or result.fun < best_value:
            x_opt = result.x[:n]
            y_opt = result.x[n:2*n]
            r_opt = result.x[2*n:3*n]
            
            circles_opt = np.column_stack([x_opt, y_opt, r_opt])
            val = _compute_value(circles_opt)
            
            if val > best_value:
                best_value = val
                best_result = circles_opt
    
    if best_result is not None:
        return best_result
    return circles_init

def _construct_simple_grid():
    """Fallback simple grid construction."""
    # 5x5 grid, take 21 circles
    side = 5
    circles = []
    for i in range(side):
        for j in range(side):
            if len(circles) >= N:
                break
            px = (i + 0.5) / side
            py = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            circles.append([px, py, r])
    
    return np.array(circles)

def construct():
    """Construct 21 disjoint circles in a perimeter-4 rectangle (unit square)."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with grid-based initialization
    circles = _construct_grid_based()
    
    # Check if we have time for optimization
    if time.time() < deadline - 20:
        try:
            circles_opt = _optimize_circles(circles, deadline)
            val_opt = _compute_value(circles_opt)
            val_init = _compute_value(circles)
            
            if val_opt > val_init:
                circles = circles_opt
        except Exception:
            pass
    
    # Final validation and cleanup
    val = _compute_value(circles)
    if val < 0:
        # Fallback to simple grid
        circles = _construct_simple_grid()
        val = _compute_value(circles)
    
    return circles

# EVOLVE-BLOCK-END
