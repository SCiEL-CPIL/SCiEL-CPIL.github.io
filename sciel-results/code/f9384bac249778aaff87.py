"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random


def construct():
    """
    Pack 21 circles in a rectangle of perimeter 4 to maximize sum of radii.
    
    Approach:
    1. Start with a grid-based initialization.
    2. Use gradient-free optimization (Nelder-Mead) to adjust circle positions 
       and radii while maintaining non-overlap constraints.
    3. The rectangle dimensions are also optimized to maximize the sum of radii.
    """
    random.seed(42)
    
    # Initial rectangle dimensions: start with square
    w0, h0 = 1.0, 1.0  # perimeter = 2*(w+h) = 4
    
    # Initialize circles in a grid pattern
    side = int(np.ceil(np.sqrt(N)))
    # We'll use a grid of side x side, but only N circles
    # Place them initially in the rectangle
    init_circles = []
    for i in range(N):
        row = i // side
        col = i % side
        x = (col + 0.5) / side * w0
        y = (row + 0.5) / side * h0
        r = min(w0, h0) / (2 * side) * 0.9  # initial radius with small margin
        init_circles.extend([x, y, r])
    
    x0 = np.array(init_circles)
    
    def objective(params):
        """
        Negative of sum of radii (we minimize negative to maximize).
        params: [w, h, x1, y1, r1, x2, y2, r2, ...]
        """
        w = params[0]
        h = params[1]
        
        # Enforce perimeter constraint: 2*(w+h) <= 4 => w+h <= 2
        # If violated, add penalty
        if w <= 0 or h <= 0:
            return 1e10
        if w + h > 2 + 1e-9:
            # Scale down to satisfy constraint
            scale = 2.0 / (w + h)
            w *= scale
            h *= scale
            if w <= 0 or h <= 0:
                return 1e10
        
        circles = params[2:].reshape((-1, 3))
        
        # Check if all circles are within the rectangle
        for i in range(N):
            cx, cy, cr = circles[i]
            if cr <= 0:
                return 1e10
            if cx - cr < -1e-9 or cx + cr > w + 1e-9 or cy - cr < -1e-9 or cy + cr > h + 1e-9:
                # Circle outside rectangle: penalty
                return 1e10 + abs(cx - cr) + abs(cx + cr - w) + abs(cy - cr) + abs(cy + cr - h)
        
        # Check for overlaps
        for i in range(N):
            for j in range(i + 1, N):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                if dist < ri + rj - 1e-9:
                    # Overlap: penalty
                    return 1e10 + (ri + rj - dist)
        
        return -np.sum(circles[:, 2])
    
    # Use Nelder-Mead optimization
    # We'll do multiple restarts with different initial conditions
    best_params = None
    best_value = np.inf
    
    for restart in range(5):
        if restart == 0:
            # First restart: grid initialization
            w_init = 1.0
            h_init = 1.0
            init = [w_init, h_init]
            for i in range(N):
                row = i // side
                col = i % side
                x = (col + 0.5) / side * w_init
                y = (row + 0.5) / side * h_init
                r = min(w_init, h_init) / (2 * side) * 0.85
                init.extend([x, y, r])
        else:
            # Subsequent restarts: add some randomness
            w_init = 0.8 + 0.4 * random.random()
            h_init = 2.0 - w_init - 0.2 * random.random()
            if h_init <= 0.2:
                h_init = 0.2
                w_init = 2.0 - h_init
            init = [w_init, h_init]
            for i in range(N):
                x = random.uniform(0, w_init)
                y = random.uniform(0, h_init)
                r = min(w_init, h_init) / (2 * side) * 0.7
                init.extend([x, y, r])
        
        result = minimize(objective, np.array(init), method='Nelder-Mead',
                         options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if result.fun < best_value:
            best_value = result.fun
            best_params = result.x
    
    if best_params is None:
        # Fallback: return a valid grid packing
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    w_opt = best_params[0]
    h_opt = best_params[1]
    circles = best_params[2:].reshape((-1, 3))
    
    # Final validation and adjustment
    # Ensure perimeter constraint is satisfied
    if w_opt + h_opt > 2.0:
        scale = 2.0 / (w_opt + h_opt)
        w_opt *= scale
        h_opt *= scale
        circles[:, 0] *= scale
        circles[:, 1] *= scale
        circles[:, 2] *= scale
    
    # Ensure circles are within bounds and non-overlapping
    # Do a final cleanup pass
    for _ in range(10):
        valid = True
        for i in range(N):
            cx, cy, cr = circles[i]
            # Clamp to rectangle
            cr = max(cr, 1e-6)
            cx = min(max(cx, cr), w_opt - cr)
            cy = min(max(cy, cr), h_opt - cr)
            circles[i] = [cx, cy, cr]
        
        # Check overlaps and resolve by shrinking
        for i in range(N):
            for j in range(i + 1, N):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                min_dist = ri + rj
                if dist < min_dist:
                    # Shrink both circles slightly
                    excess = min_dist - dist
                    factor = 1.0 - 0.01 * excess
                    ri *= max(factor, 0.5)
                    rj *= max(factor, 0.5)
                    circles[i, 2] = ri
                    circles[j, 2] = rj
                    valid = False
        if valid:
            break
    
    return circles
# EVOLVE-BLOCK-END
