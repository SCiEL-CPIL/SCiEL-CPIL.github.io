"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _penalty(x, w, h):
    """Compute constraint violation for circle packing."""
    n = len(x)
    penalty = 0.0
    # Boundary constraints
    for i in range(n):
        cx, cy, r = x[i]
        if cx - r < 0:
            penalty += (cx - r) ** 2
        if cy - r < 0:
            penalty += (cy - r) ** 2
        if cx + r > w:
            penalty += (cx + r - w) ** 2
        if cy + r > h:
            penalty += (cy + r - h) ** 2
    # Pairwise distance constraints
    for i in range(n):
        for j in range(i + 1, n):
            cx1, cy1, r1 = x[i]
            cx2, cy2, r2 = x[j]
            dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
            min_dist = r1 + r2
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2
    return penalty

def _objective(x, w, h):
    """Objective: negative sum of radii plus penalty."""
    n = len(x)
    radii = x[2::3]
    sum_r = np.sum(radii)
    pen = _penalty(x, w, h)
    return -sum_r + 100 * pen

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    best_solution = None
    best_value = -np.inf
    
    # Try multiple aspect ratios for the rectangle
    # Perimeter = 2*(w+h) = 4, so w+h = 2
    aspect_ratios = [1.0, 1.2, 1.5, 2.0, 0.8, 0.6, 0.5, 1.3, 1.8]
    
    for aspect in aspect_ratios:
        if time.time() > deadline - 5:
            break
            
        # w + h = 2, h/w = 1/aspect => w*(1 + 1/aspect) = 2
        w = 2.0 / (1 + 1/aspect)
        h = 2.0 - w
        
        if w <= 0 or h <= 0:
            continue
        
        # Initialize with grid-like positions
        n = N
        side = int(np.ceil(np.sqrt(n)))
        x0 = []
        for i in range(side):
            for j in range(side):
                if len(x0) >= n:
                    break
                cx = (i + 0.5) * w / side
                cy = (j + 0.5) * h / side
                r = min(w, h) / (2 * side) * 0.9
                x0.extend([cx, cy, r])
        
        x0 = np.array(x0[:n * 3])
        
        # Run optimization
        try:
            result = minimize(
                _objective,
                x0,
                args=(w, h),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if result.success or result.fun < best_value:
                sol = result.x
                # Verify constraints
                pen = _penalty(sol, w, h)
                if pen < 1e-6:
                    value = np.sum(sol[2::3])
                    if value > best_value:
                        best_value = value
                        best_solution = sol
        except Exception:
            pass
        
        if time.time() > deadline - 5:
            break
    
    # If no valid solution found, fall back to simple grid
    if best_solution is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Format output: (cx, cy, r) for each circle
    result = np.array(best_solution).reshape(-1, 3)
    return result
# EVOLVE-BLOCK-END
