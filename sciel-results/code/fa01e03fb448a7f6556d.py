"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def construct():
    """Pack 21 circles in a perimeter-4 rectangle to maximize sum of radii."""
    t_start = time.time()
    deadline = t_start + 110.0  # safety margin before 120s limit

    # We'll use a gradient-based / local search approach.
    # Start with a grid arrangement, then optimize positions and radii.
    
    # Initial guess: 5x5 grid (25 cells, use 21)
    # Rectangle with perimeter 4: w + h = 2
    # Let's parameterize by aspect ratio or just fix a reasonable shape.
    
    # For 21 circles, a good layout might be roughly 5 wide x 4-5 tall
    # Let's try to optimize both the rectangle dimensions and circle positions/radii.
    
    # Strategy: 
    # 1. Start with equal radii in a grid
    # 2. Use scipy.optimize to maximize sum of radii subject to constraints
    #    - No overlap: dist(i,j) >= r_i + r_j
    #    - Boundary: r_i <= x_i, r_i <= y_i, r_i <= w-x_i, r_i <= h-y_i
    #    - Perimeter: 2*(w+h) = 4 => w+h = 2
    #    - All radii > 0
    
    # Since scipy.optimize.minimize with constraints can be slow, 
    # let's try a simpler iterative approach:
    # - Fix aspect ratio initially, optimize positions and radii
    # - Or use a penalty method
    
    # Let's use scipy.optimize with SLSQP or trust-constr
    from scipy.optimize import minimize
    
    # Variables: [w, x_0, y_0, r_0, x_1, y_1, r_1, ..., x_20, y_20, r_20]
    # w is width, h = 2 - w
    # Actually, let's just optimize positions and radii for a fixed aspect ratio,
    # then try a few aspect ratios.
    
    best_val = 0.0
    best_result = None
    
    # Try several aspect ratios
    # w + h = 2, so w in (0, 2)
    # For 21 circles, roughly square-ish might be good, or slightly rectangular
    aspect_ratios = [1.0, 1.1, 0.9, 1.2, 0.85, 1.15, 0.8, 1.25]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 5:
            break
        
        # w/h = ar, w + h = 2
        # w = ar * h, ar*h + h = 2, h = 2/(ar+1), w = 2*ar/(ar+1)
        h = 2.0 / (ar + 1.0)
        w = 2.0 * ar / (ar + 1.0)
        
        # Initial grid placement
        # 21 circles: try 5x5 grid with 4 removed, or 7x3, etc.
        # Let's use a 5x5 grid centered, take first 21
        cols = 5
        rows = 5
        # Positions in [0, w] x [0, h]
        xs = np.linspace(0.1*w, 0.9*w, cols)
        ys = np.linspace(0.1*h, 0.9*h, rows)
        
        # Create grid positions
        pos = []
        for i in range(cols):
            for j in range(rows):
                pos.append([xs[i], ys[j]])
        pos = np.array(pos[:21])
        
        # Initial radii: estimate from spacing
        dx = xs[1] - xs[0] if cols > 1 else w
        dy = ys[1] - ys[0] if rows > 1 else h
        r_init = min(dx, dy) * 0.45
        
        # Variables: x_0, y_0, r_0, x_1, y_1, r_1, ...
        # 21 circles * 3 = 63 variables
        x0 = np.zeros(63)
        for i in range(21):
            x0[3*i] = pos[i][0]
            x0[3*i+1] = pos[i][1]
            x0[3*i+2] = r_init
        
        # Objective: maximize sum of radii => minimize -sum(r_i)
        def objective(v):
            return -sum(v[3*i+2] for i in range(21))
        
        # Constraints:
        # 1. Boundary: x_i >= r_i, x_i <= w - r_i, y_i >= r_i, y_i <= h - r_i
        # 2. No overlap: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        
        # For SLSQP, we need inequality constraints: g(x) >= 0
        # Boundary: x_i - r_i >= 0, w - x_i - r_i >= 0, y_i - r_i >= 0, h - y_i - r_i >= 0
        # Overlap: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        
        # This is a large number of constraints. Let's use a penalty approach instead
        # for speed, or use trust-constr.
        
        # Let's try a simpler iterative local optimization:
        # Start from grid, then do gradient ascent on sum of radii with penalty for violations.
        
        def penalized_obj(v):
            """Negative of (sum radii - penalty for violations)."""
            r = 0.0
            penalty = 0.0
            for i in range(21):
                xi = v[3*i]
                yi = v[3*i+1]
                ri = v[3*i+2]
                r += ri
                # Boundary violations
                if xi < ri:
                    penalty += (xi - ri) ** 2
                if w - xi < ri:
                    penalty += (w - xi - ri) ** 2
                if yi < ri:
                    penalty += (yi - ri) ** 2
                if h - yi < ri:
                    penalty += (h - yi - ri) ** 2
            # Overlap violations
            for i in range(21):
                for j in range(i+1, 21):
                    xi = v[3*i]
                    yi = v[3*i+1]
                    ri = v[3*i+2]
                    xj = v[3*j]
                    yj = v[3*j+1]
                    rj = v[3*j+2]
                    d2 = (xi - xj)**2 + (yi - yj)**2
                    rsum = ri + rj
                    if d2 < rsum**2:
                        penalty += (rsum**2 - d2) ** 2
            
            return -(r - 1000 * penalty)
        
        # Use Nelder-Mead for the penalized objective
        # This might be slow for 63 dimensions. Let's try a smaller search.
        
        # Alternative: fix radii to be equal, optimize positions only, then adjust radii.
        # Or: use a coordinate descent approach.
        
        # Let's try scipy.optimize.minimize with BFGS on the penalized objective
        result = minimize(penalized_obj, x0, method='BFGS', 
                          options={'maxiter': 200, 'gtol': 1e-4})
        
        if result.success or result.fun < 0:
            v = result.x
            # Check validity
            valid = True
            for i in range(21):
                xi = v[3*i]
                yi = v[3*i+1]
                ri = v[3*i+2]
                if xi < ri - 1e-6 or w - xi < ri - 1e-6 or yi < ri - 1e-6 or h - yi < ri - 1e-6:
                    valid = False
                    break
                if ri <= 0:
                    valid = False
                    break
            if valid:
                for i in range(21):
                    for j in range(i+1, 21):
                        xi = v[3*i]
                        yi = v[3*i+1]
                        ri = v[3*i+2]
                        xj = v[3*j]
                        yj = v[3*j+1]
                        rj = v[3*j+2]
                        d2 = (xi - xj)**2 + (yi - yj)**2
                        if d2 < (ri + rj)**2 - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                total_r = sum(v[3*i+2] for i in range(21))
                if total_r > best_val:
                    best_val = total_r
                    best_result = v.copy()
                    w_best = w
                    h_best = h
    
    # If we found a good result, return it
    if best_result is not None:
        result_arr = np.zeros((21, 3))
        for i in range(21):
            result_arr[i][0] = best_result[3*i]
            result_arr[i][1] = best_result[3*i+1]
            result_arr[i][2] = best_result[3*i+2]
        return result_arr
    
    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
