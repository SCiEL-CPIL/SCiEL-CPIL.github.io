"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import sys

def _penalty(circles):
    """Compute penalty for constraint violations. circles: (n,3) array [cx, cy, r]."""
    cx, cy, r = circles[:, 0], circles[:, 1], circles[:, 2]
    # Bounding box
    x_min, x_max = cx.min() - r, cx.max() + r
    y_min, y_max = cy.min() - r, cy.max() + r
    W = x_max - x_min
    H = y_max - y_min
    perim = 2.0 * (W + H)
    perim_pen = max(0.0, perim - 4.0 + 1e-9) ** 2 * 100.0
    
    # Non-overlap: dist(i,j) >= r_i + r_j
    n = len(circles)
    overlap_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((cx[i] - cx[j])**2 + (cy[i] - cy[j])**2)
            min_d = r[i] + r[j]
            if d < min_d:
                overlap_pen += (min_d - d) ** 2 * 50.0
    return perim_pen + overlap_pen

def _objective(circles):
    """Objective: negative sum of radii + penalties."""
    r = circles[:, 2]
    return -np.sum(r) + _penalty(circles)

def _is_valid(circles, tol=1e-7):
    """Check if configuration is valid."""
    cx, cy, r = circles[:, 0], circles[:, 1], circles[:, 2]
    if np.any(r < -tol):
        return False
    x_min, x_max = cx.min() - r, cx.max() + r
    y_min, y_max = cy.min() - r, cy.max() + r
    W = x_max - x_min
    H = y_max - y_min
    perim = 2.0 * (W + H)
    if perim > 4.0 + 1e-6:
        return False
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((cx[i] - cx[j])**2 + (cy[i] - cy[j])**2)
            if d < r[i] + r[j] - 1e-6:
                return False
    return True

def _make_initial(side, rng):
    """Create initial configuration in a grid with jitter."""
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side + (rng.random() - 0.5) * 0.1 / side
            cy = (j + 0.5) / side + (rng.random() - 0.5) * 0.1 / side
            r = 0.5 / side * 0.95
            cells.append([cx, cy, r])
    return np.array(cells[:N])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    best_val = -np.inf
    best_circles = None
    
    # Seed for reproducibility
    rng = np.random.RandomState(42)
    
    # Strategy: multiple restarts with different initial configurations
    # Use Nelder-Mead or L-BFGS-B for optimization
    
    attempts = 0
    max_attempts = 200
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Choose a grid configuration
        # For 21 circles, try 5x5 grid (25 cells, use 21)
        # Or other configurations
        if attempts <= 50:
            side = 5
        elif attempts <= 100:
            side = 5  # Still 5x5 but different jitter
        else:
            # Try rectangular grids
            if attempts <= 130:
                side = 5
            else:
                side = 5
        
        circles = _make_initial(side, rng)
        
        # Normalize: center the configuration
        cx, cy = circles[:, 0], circles[:, 1]
        cx_center = (cx.min() + cx.max()) / 2
        cy_center = (cy.min() + cy.max()) / 2
        circles[:, 0] -= cx_center
        circles[:, 1] -= cy_center
        
        # Optimize using Nelder-Mead on flattened vector
        x0 = circles.flatten()
        
        def obj_flat(x):
            c = x.reshape(N, 3)
            return _objective(c)
        
        try:
            result = minimize(obj_flat, x0, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
        except Exception:
            continue
        
        circles_opt = result.x.reshape(N, 3)
        
        # Check validity
        if _is_valid(circles_opt):
            val = np.sum(circles_opt[:, 2])
            if val > best_val:
                best_val = val
                best_circles = circles_opt.copy()
        
        # Also check the initial configuration
        if _is_valid(circles):
            val = np.sum(circles[:, 2])
            if val > best_val:
                best_val = val
                best_circles = circles.copy()
        
        # Periodic check
        if time.time() > deadline:
            break
    
    # If no valid configuration found, fall back to seed
    if best_circles is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
    
    # Final validation and cleanup
    cx, cy, r = best_circles[:, 0], best_circles[:, 1], best_circles[:, 2]
    x_min, x_max = cx.min() - r, cx.max() + r
    y_min, y_max = cy.min() - r, cy.max() + r
    W = x_max - x_min
    H = y_max - y_min
    perim = 2.0 * (W + H)
    
    # If perimeter is slightly over, scale down
    if perim > 4.0:
        scale = 4.0 / perim
        # Scale positions and radii from center
        cx_center = (x_min + x_max) / 2
        cy_center = (y_min + y_max) / 2
        best_circles[:, 0] = cx_center + (best_circles[:, 0] - cx_center) * scale
        best_circles[:, 1] = cy_center + (best_circles[:, 1] - cy_center) * scale
        best_circles[:, 2] *= scale
    
    return best_circles
# EVOLVE-BLOCK-END
